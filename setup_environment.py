# setup_environment.py
import os
import subprocess
import sys
import logging
import site

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
VENV_DIR = ".\\venv"
LLAMA_CPP_DIR = ".\\llama_cpp"
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
    if not os.path.exists(LLAMA_CPP_DIR):
        logger.info(f"Cloning llama.cpp to {LLAMA_CPP_DIR}")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone llama.cpp: {str(e)}")
            sys.exit(1)
    
    # 构建llama.cpp
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    logger.info("Building llama.cpp")
    try:
        subprocess.run(["cmake", "..", "-DLLAMA_CUDA=ON"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build llama.cpp: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    logger.info("Starting environment setup")
    created = create_venv()
    install_dependencies()
    setup_llama_cpp()
    logger.info("Environment setup completed successfully")

if __name__ == "__main__":
    main()