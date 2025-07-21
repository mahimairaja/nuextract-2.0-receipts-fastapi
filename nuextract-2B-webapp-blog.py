import fastapi
import fastapi.staticfiles
import modal

# Create a Modal app instance
app = modal.App("nuextract-2.0-webapp")
web_app = fastapi.FastAPI()


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    # Reference the remote Modal function for parsing receipts
    parse_receipt = modal.Function.from_name("nuextract-2.0-function", "parse_receipt")

    form = await request.form()
    receipt = await form["receipt"].read()  # type: ignore
    call = parse_receipt.spawn(receipt)  # Asynchronously invoke the function
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    # Retrieve the function call by ID
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)  # Try to get result immediately
    except TimeoutError:
        return fastapi.responses.JSONResponse(
            content="", status_code=202
        )  # Still processing

    return result


# Define the image for the Modal app
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4"
)


@app.function(image=image)
@modal.asgi_app()
def wrapper():
    # Entrypoint for the Modal ASGI app
    return web_app
