import io
import requests
import gradio as gr

from PIL import Image
from style import style

CLASSIFICATION_SERVICE_URL = "http://classification-service:8000/predict"
EXPLAINABILITY_SERVICE_URL = "http://explainability-service:8000/explain"
HISTORY_SERVICE_URL = "http://history-service:8000/logs"

def update_logs():
    try:
        response = requests.get(HISTORY_SERVICE_URL)
        if response.status_code == 200:
            raw_data = response.json()
            data = [
                [item['date_time'], item['predicted_class'], item['confidence']]
                for item in raw_data
            ]
            return data
        return []
    
    except Exception:
        return []

def diagnose(slice : Image):
    if slice is None:
        return None, None
    
    image_bytes = io.BytesIO()
    slice.save(image_bytes, format = 'PNG')

    try:
        image_bytes.seek(0)      
        files = { 'file': ('image.png', image_bytes, 'image/png') }

        response = requests.post(CLASSIFICATION_SERVICE_URL, files = files)
        response.raise_for_status()

        predictions = response.json()['all_predictions']

        try:
            diagnosis_log = {
                "predicted_class": response.json()['predicted_class'],
                "confidence": response.json()['confidence']
            }
            requests.post(HISTORY_SERVICE_URL, json = diagnosis_log)

        except Exception as e:
            print(f"Unable to save diagnosis log: {e}")

        image_bytes.seek(0)      
        files = { 'file': ('image.png', image_bytes, 'image/png') }

        response = requests.post(EXPLAINABILITY_SERVICE_URL, files = files)
        response.raise_for_status()

        grad_cam = Image.open(io.BytesIO(response.content)) 

        return predictions, grad_cam    

    except requests.exceptions.RequestException as e:
        msg = {"Connection error": 1.0}
        
        return msg, None
    
with gr.Blocks() as app:
    with gr.Column():
        gr.Markdown(
            """
            ## Classificazione Tumori Cerebrali
            Sviluppato da **Giacomo Liguori**            
            """,
            elem_classes = 'center'
        )
        with gr.Column():
            with gr.Row(equal_height = True):
                with gr.Column():
                    slice_input = gr.Image(type = 'pil', label = "Risonanza Magnetica")
                    submit_input = gr.Button(value = "Elabora Referto", variant = 'primary')
                    refresh_table_logs = gr.Button(value = "Aggiorna Storico", variant = 'secondary')
                with gr.Column():
                    logits = gr.Label(num_top_classes = 3, label = "Classe Tumorale")
                    gradients = gr.Image(type = 'pil', label = "Camera Gradienti")
            with gr.Row():
                logs = gr.Dataframe(
                    headers = ["Quando", "Tumore Diagnosticato", "Affidabilita"],
                    datatype = ["str", "str", "str"],
                    label = "Storico Diagnosi",
                    interactive = False,
                    wrap = True
                )

    submit_input.click(
        fn = diagnose,
        inputs = slice_input, 
        outputs = [logits, gradients]
    )
    refresh_table_logs.click(
        fn = update_logs,
        inputs = None,
        outputs = logs
    )

if __name__ == "__main__":
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        theme = gr.Theme.from_hub("hmb/spark"), css = style
        )