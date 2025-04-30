# convert_to_gguf.py
import os
import subprocess
import logging

# ������־
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ���ó���
LLAMA_CPP_DIR = ".\\llama_cpp"
GGUF_OUTPUT_DIR = ".\\gguf_model"
QUANTIZATION_METHOD = "q4_k_m"
IMATRIX_DATA = ".\\imatrix_data.txt"
MODEL_DIR = ".\\fine_tuned_model"

def convert_to_gguf():
    """ת��ģ��ΪGGUF��ʽ"""
    if not os.path.exists(GGUF_OUTPUT_DIR):
        os.makedirs(GGUF_OUTPUT_DIR)
    
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "quantize.exe")
    imatrix_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "imatrix.exe")
    
    # 1. ת��ΪFP16
    fp16_file = os.path.join(GGUF_OUTPUT_DIR, "model-fp16.bin")
    logger.info("Converting to FP16")
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert.py")
    subprocess.run([
        "python", convert_script, MODEL_DIR, 
        "--outfile", fp16_file, 
        "--outtype", "f16"
    ], check=True)
    
    # 2. ����imatrix
    imatrix_output = os.path.join(GGUF_OUTPUT_DIR, "imatrix.dat")
    logger.info("Generating imatrix")
    subprocess.run([
        imatrix_bin, "-m", fp16_file,
        "-f", IMATRIX_DATA,
        "-o", imatrix_output
    ], check=True)
    
    # 3. ����ģ��
    gguf_file = os.path.join(GGUF_OUTPUT_DIR, f"model-{QUANTIZATION_METHOD}.gguf")
    logger.info(f"Quantizing to {QUANTIZATION_METHOD}")
    subprocess.run([
        quantize_bin, fp16_file,
        gguf_file, QUANTIZATION_METHOD,
        "--imatrix", imatrix_output
    ], check=True)
    
    logger.info(f"GGUF model saved to {gguf_file}")

def main():
    """������"""
    logger.info("Starting GGUF conversion")
    convert_to_gguf()
    logger.info("GGUF conversion completed")

if __name__ == "__main__":
    main()