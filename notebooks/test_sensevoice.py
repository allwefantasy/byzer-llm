import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.byzerllm.sensevoice import init_model, format_str_v3
import base64
import json


def main():
    # Initialize the model
    model_dir = "/Users/allwefantasy/models/SenseVoiceSmall"
    model, _ = init_model(model_dir)

    # Prepare test audio
    with open(
        "/Users/allwefantasy/models/SenseVoiceSmall/example/zh.mp3", "rb"
    ) as audio_file:
        audio_data = audio_file.read()
        base64_audio = base64.b64encode(audio_data).decode("utf-8")        
        # Prepare input in the expected format
        test_input = json.dumps(
            {
                            "type": "audio",
                            "audio": f"data:audio/wav;base64,{base64_audio}",
                        }
        )

        # Call stream_chat
        result = model.stream_chat(tokenizer=None, ins = test_input)

        # Process and print the result
        if result and len(result) > 0:
            output, metadata = result[0]
            parsed_output = json.loads(output)
            formatted_text = parsed_output.get("text", "")
            print("Transcription:", formatted_text)
            print("Metadata:", metadata)
        else:
            print("No result returned from the model.")


if __name__ == "__main__":
    main()
