"""Create a web application that 
predicts the viability of a Kickstarter
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Instantiate fastAPI with appropriate descriptors

app = FastAPI(
    title="Kickstarter Success Guide",
    description="Interactive tool to check for the success of a Kickstarter",
    version="1.0",
    docs_url="/docs"
)

# Instantiate templates path
templates = Jinja2Templates(directory="/templates")


# Route for home page
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    """Displays index.html from templateswhen user loads home URL"""
    return templates.TemplateResponse("index.html", {"request": request})


# Route for the about page
@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# Predictive model
app.mount(
    "/model.kickstarterlib",
    StaticFiles(directory="/ML"),
    name="model.kickstarterlib"
    )
