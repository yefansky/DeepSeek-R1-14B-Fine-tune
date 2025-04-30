# setup_environment.py
import os
import subprocess
import sys
import logging
import urllib.request
import zipfile
import shutil
import tarfile

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["PYTHONUTF8"] = "1"  # 确保子进程使用 UTF-8 编码

# 配置常量
VENV_DIR = ".\\venv"
LLAMA_CPP_DIR = ".\\llama_cpp"
CURL_DIR_INCLUDE = ".\\curl-8.13.0\\include"
CURL_DIR_LIB = ".\\curl_build\\lib\\Release"

REQUIREMENTS = [
    "unsloth",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "bitsandbytes",
    "huggingface_hub",
    "flask",
    "gitpython",
    "torch"
]

def create_venv():
    """创建虚拟环境"""
    if not os.path.exists(VENV_DIR):
        logger.info(f"Creating virtual environment at {VENV_DIR}")
        try:
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            logger.info(f"Virtual environment created at {VENV_DIR}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {str(e)}")
            sys.exit(1)
    return False

def upgrade_pip():
    """升级虚拟环境中的pip"""
    venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    if not os.path.exists(venv_pip):
        logger.error("Virtual environment pip not found")
        sys.exit(1)
    
    logger.info("Upgrading pip")
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upgrade pip: {str(e)}")
        sys.exit(1)

def install_dependencies():
    """在虚拟环境中安装依赖"""
    venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    if not os.path.exists(venv_pip):
        logger.error("Virtual environment pip not found")
        sys.exit(1)
    
    for package in REQUIREMENTS:
        logger.info(f"Installing {package}")
        try:
            subprocess.run([venv_pip, "install", package], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {str(e)}")
            sys.exit(1)

def build_curl_lib():
    # 定义变量
    url = "https://curl.se/download/curl-8.13.0.tar.xz"
    filename = "curl-8.13.0.tar.xz"
    extract_dir = "curl-8.13.0"
    build_dir = "curl_build"
    
    try:
        # 1. 下载文件
        print("Downloading curl...")
        urllib.request.urlretrieve(url, filename)
        
        # 2. 解包
        print("Extracting tar.xz...")
        with tarfile.open(filename, "r:xz") as tar:
            tar.extractall()
        
        # 3. 创建构建目录
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        os.chdir(build_dir)
        
        # 4. 运行CMake配置，禁用所有问题依赖
        print("Running CMake...")
        cmake_cmd = [
            "cmake",
            f"../{extract_dir}",
            "-DBUILD_SHARED_LIBS=OFF",  # 静态库
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCURL_DISABLE_LIBIDN2=ON",  # 禁用libidn2
            "-DCURL_USE_LIBPSL=OFF",     # 明确禁用libpsl
            "-DCURL_DISABLE_CRYPTO_AUTH=ON",  # 禁用需要OpenSSL等的加密功能
            "-DCURL_DISABLE_ZLIB=ON",    # 禁用zlib
            "-DCURL_DISABLE_BROTLI=ON",  # 禁用brotli
            "-DCURL_DISABLE_ZSTD=ON",    # 禁用zstd
            "-DCURL_DISABLE_NGHTTP2=ON", # 禁用nghttp2
            "-DBUILD_CURL_EXE=OFF",      # 不构建curl可执行文件
            "-DBUILD_TESTING=OFF"        # 禁用测试
        ]
        subprocess.run(cmake_cmd, check=True)
        
        # 5. 编译
        print("Building curl...")
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
        subprocess.run(build_cmd, check=True)
        
        print("Build completed! libcurl.lib should be in curl_build/lib/Release/")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
    finally:
        # 清理：返回原始目录并删除下载的tar文件
        os.chdir("..")
        if os.path.exists(filename):
            os.remove(filename)            

def setup_llama_cpp():
    """设置llama.cpp"""

    build_curl_lib()

    if not os.path.exists(LLAMA_CPP_DIR):
        logger.info(f"Cloning llama.cpp to {LLAMA_CPP_DIR}")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone llama.cpp: {str(e)}")
            sys.exit(1)
    
    # 设置 CURL 路径（使用绝对路径）
    curl_include_dir = os.path.abspath(os.path.join(CURL_DIR_INCLUDE))
    curl_library = os.path.abspath(os.path.join(CURL_DIR_LIB))
    
    # 验证路径
    if not os.path.exists(curl_include_dir):
        logger.error(f"CURL include directory ({curl_include_dir}) does not exist")
        sys.exit(1)
    if not os.path.exists(curl_library):
        logger.error(f"CURL library ({curl_library}) does not exist")
        sys.exit(1)

    # 构建llama.cpp
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    logger.info("Building llama.cpp")
    try:
        subprocess.run(["cmake", "..", 
                        "-DGGML_CUDA=ON", 
                        "-DCMAKE_GENERATOR_TOOLSET='cuda=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8'",
                        f"-DCURL_INCLUDE_DIR={curl_include_dir}",
                        f"-DCURL_LIBRARY={curl_library}"
                        ], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build llama.cpp: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    logger.info("Starting environment setup")
    created = create_venv()
    upgrade_pip()  # 在创建新虚拟环境后升级pip
    install_dependencies()
    setup_llama_cpp()
    logger.info("Environment setup completed successfully")

if __name__ == "__main__":
    main()