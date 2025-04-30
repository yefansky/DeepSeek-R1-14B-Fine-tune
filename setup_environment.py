# setup_environment.py
import os
import subprocess
import sys
import logging
import urllib.request
import zipfile
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
VENV_DIR = ".\\venv"
LLAMA_CPP_DIR = ".\\llama_cpp"
CURL_DIR = ".\\curl"
CURL_URL = "https://curl.se/windows/latest.cgi?p=win64-mingw.zip"

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

def setup_llama_cpp():
    """设置llama.cpp"""

    download_and_extract_curl()

    if not os.path.exists(LLAMA_CPP_DIR):
        logger.info(f"Cloning llama.cpp to {LLAMA_CPP_DIR}")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone llama.cpp: {str(e)}")
            sys.exit(1)
    
    # 设置 CURL 路径（使用绝对路径）
    curl_include_dir = os.path.abspath(os.path.join(CURL_DIR, "include"))
    curl_library = os.path.abspath(os.path.join(CURL_DIR, "lib"))
    
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

def download_and_extract_curl():
    """下载并解压 CURL 到指定目录，并整理目录结构"""
    curl_include = os.path.join(CURL_DIR, "include", "curl", "curl.h")
    curl_library = os.path.join(CURL_DIR, "lib", "libcurl.a")
    
    if os.path.exists(curl_include) and os.path.exists(curl_library):
        logger.info(f"CURL directory {CURL_DIR} already contains required files, skipping download")
        return
    
    logger.info(f"Downloading CURL from {CURL_URL}")
    curl_zip = "curl.zip"
    try:
        urllib.request.urlretrieve(CURL_URL, curl_zip)
        logger.info("CURL downloaded successfully")
        
        # 确保 CURL_DIR 存在
        if not os.path.exists(CURL_DIR):
            os.makedirs(CURL_DIR)
        
        # 解压 ZIP 文件
        with zipfile.ZipFile(curl_zip, 'r') as zip_ref:
            zip_ref.extractall(CURL_DIR)
        logger.info(f"CURL extracted to {CURL_DIR}")
        
        # 删除 ZIP 文件
        os.remove(curl_zip)
        
        # 查找解压后的子目录（形如 curl-x.y.z-win64）
        sub_dir = None
        for item in os.listdir(CURL_DIR):
            if item.startswith("curl-") and os.path.isdir(os.path.join(CURL_DIR, item)):
                sub_dir = os.path.join(CURL_DIR, item)
                break
        
        if sub_dir:
            # 将子目录内容移动到 CURL_DIR 根目录
            for item in os.listdir(sub_dir):
                src = os.path.join(sub_dir, item)
                dst = os.path.join(CURL_DIR, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(sub_dir)
            logger.info(f"Moved contents from {sub_dir} to {CURL_DIR}")
        else:
            logger.error("Expected subdirectory (e.g., curl-x.y.z-win64) not found")
            sys.exit(1)
            
        # 验证必要文件
        if not os.path.exists(curl_include):
            logger.error(f"CURL header file ({curl_include}) not found after extraction")
            sys.exit(1)
        if not os.path.exists(curl_library):
            logger.error(f"CURL library file ({curl_library}) not found after extraction")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download or extract CURL: {str(e)}")
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