# Author: Sakthi Santhosh
# Created on: 15/04/2023
def main(argv) -> int:
    if len(argv) < 1:
        print("Error: Program called with no data.")
        return 1

    import os.path

    if not os.path.exists(argv[0]):
        print("Error: Image file not found.")
        return 1

    from numpy import array, expand_dims, float32
    from PIL import Image
    from tflite_runtime.interpreter import Interpreter
    from time import time

    processed_image = expand_dims(
        array(
            Image.open(argv[0]).convert("RGB").resize((256, 256)),
            dtype=float32
        ) / 255, axis=0
    )

    interpreter = Interpreter("./model.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], processed_image)

    start_time = time()
    interpreter.invoke()
    time_delta = (time() - start_time) * 1000
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    print("Inference time: %d ms"%(time_delta))
    print("Prediction:", predictions)
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
