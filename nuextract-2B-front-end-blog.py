import time

import gradio as gr
import requests

# Default backend URL for Modal deployment
DEFAULT_BACKEND_URL = "PASTE_YOUR_DEPLOYED_URL_HERE"


def extract_text(receipt, api_uri, progress=gr.Progress(track_tqdm=True)):
    if receipt is None:
        return gr.update(value="Please upload a receipt image.", interactive=True)

    # Use user-provided API URI or default
    backend_url = (
        api_uri.strip() if api_uri and api_uri.strip() else DEFAULT_BACKEND_URL
    )
    parse_url = f"{backend_url}/parse"
    result_url = f"{backend_url}/result"

    try:
        mime_type = getattr(receipt, "type", "application/octet-stream")
        with open(receipt.name, "rb") as f:
            files = {"receipt": (receipt.name, f, mime_type)}
            resp = requests.post(parse_url, files=files)  # Send image to backend
        if not resp.ok:
            return f"Error from backend: {resp.status_code} {resp.text}"
        data = resp.json()
        call_id = data.get("call_id")
        if not call_id:
            return "No call_id returned from backend."
        # Poll /result/{call_id}
        for i in range(60):  # Poll for up to 60 seconds
            poll_resp = requests.get(f"{result_url}/{call_id}")
            if poll_resp.status_code == 202:
                progress((i + 1) / 60, desc="Processing...")
                time.sleep(1)
                continue
            if poll_resp.ok:
                try:
                    result = poll_resp.json()
                    if isinstance(result, dict):
                        return result.get("result", str(result))
                    return str(result)
                except Exception:
                    return poll_resp.text
            else:
                return f"Error polling result: {poll_resp.status_code} {poll_resp.text}"
        return "Timed out waiting for OCR result."
    except Exception as e:
        return f"Exception: {e}"


# JavaScript to force dark theme on Gradio
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def main():
    with gr.Blocks(
        js=js_func, theme=gr.themes.Soft(primary_hue="orange", secondary_hue="gray")
    ) as demo:
        # Title and subtitle
        gr.HTML("""
        <style>
        @media (prefers-color-scheme: dark) {
            .elegant-ocr-title, .elegant-ocr-subtitle { color: #fff !important; }
        }
        @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
            .elegant-ocr-title { color: #222 !important; }
            .elegant-ocr-subtitle { color: #555 !important; }
        }
        </style>
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <h1 class='elegant-ocr-title' style='font-size: 2.5rem; font-weight: 700;'>Receipt OCR based on NuExtract-2.0</h1>
            <p class='elegant-ocr-subtitle' style='font-size: 1.1rem;'>Upload a receipt image and extract text using NuExtract-2.0 hosted on Modal.</p>
        </div>
        """)

        with gr.Group():
            api_uri = gr.Textbox(
                label="Backend API URI",
                placeholder="https://your-backend-url.modal.run",
                info="Please enter your backend api url",
                lines=1,
            )

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Group():
                    receipt = gr.File(
                        label="Upload Receipt Image",
                        file_types=["image"],
                        type="filepath",
                    )
                    image_preview = gr.Image(
                        label="Preview",
                        visible=False,
                        show_label=True,
                        height=250,
                        width=250,
                        elem_id="preview-img",
                    )
            with gr.Column(scale=2, min_width=400):
                with gr.Group():
                    result_box = gr.Textbox(
                        label="OCR Result",
                        lines=14,
                        interactive=True,
                        show_copy_button=True,
                        elem_id="result-box",
                        container=True,
                        max_lines=20,
                        placeholder="The extracted text will appear here...",
                    )
        extract_btn = gr.Button("Extract Text", variant="primary", size="lg")
        status = gr.Markdown("", visible=False)

        # Add examples section
        gr.Examples(
            examples=[["images/receipt.png"]],
            inputs=receipt,
            label="📄 Example Receipt",
        )

        # Show preview of uploaded image
        def show_preview(file):
            if file is not None:
                return gr.update(value=file.name, visible=True)
            return gr.update(visible=False)

        receipt.change(fn=show_preview, inputs=receipt, outputs=image_preview)
        extract_btn.click(
            fn=extract_text, inputs=[receipt, api_uri], outputs=result_box
        )

        gr.HTML("""
        <footer style='text-align: right; margin-top: 2rem; color: #888;'>
            Powered by <a href='https://modal.com' target='_blank' style='color: #007FFF; text-decoration: none; font-weight: 600;'>Mahimai AI Labs ❤️</a>
        </footer>
        """)
    demo.launch()


if __name__ == "__main__":
    main()
