import model_inference

if __name__ == "__main__":
    print("Testing model initialization...")
    model_inference.init_model()
    print("Testing inference...")
    result = model_inference.inference("/home/michael/Desktop/zidane.jpg")
    print(f"Inference result: {result}")
