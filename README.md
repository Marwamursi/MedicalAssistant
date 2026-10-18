# NeuroScan AI — Brain MRI Analysis Assistant

NeuroScan AI is an educational AI project that combines deep learning image classification with a vision-language model to generate educational reports from brain MRI images.

## Project Overview

The application allows users to upload a supported 2D brain MRI image, view experimental classification scores, generate an AI-assisted educational report, and export the results as a PDF or TXT file.

**Important:** This project is for educational and research purposes only. It is not a clinically validated diagnostic system. Model scores are not confirmed disease probabilities and must not be used to make medical decisions.

## Technologies Used

- **Python** — Main programming language
- **PyTorch** — Deep learning framework
- **MobileNetV3-Small** — MRI image classification model
- **Qwen2.5-VL 3B** — Vision-language model used for educational report generation
- **Ollama** — Runs the Qwen model locally
- **Streamlit** — Web interface
- **ReportLab** — PDF report generation
- **Docker** — Application containerization

## Classification Classes

The trained classifier predicts one of four dataset classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

These are experimental model outputs, not confirmed clinical diagnoses.

## Project Structure

```text
MedicalAssistant/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── outputs/
    ├── best_model.pth
    └── evaluation_report.json
```

## Requirements

- Docker Desktop
- Ollama installed and running on the host computer
- The `qwen2.5vl:3b` model available in Ollama

## Run with Docker

Build the Docker image from the project directory:

```bash
docker build -t medical-ai .
```

Run the application:

```bash
docker run -d --name medical-ai-app -p 8501:8501 medical-ai
```

Open the application in your browser:

```text
http://localhost:8501
```

The application uses `host.docker.internal` to connect to Ollama running on the host computer. Ensure Ollama is running and the required model is installed before generating reports.

To stop the container:

```bash
docker stop medical-ai-app
```

## Application Workflow

1. Upload a supported PNG, JPG, or JPEG brain MRI image.
2. Run the MobileNetV3-Small classifier.
3. Review the experimental classification scores.
4. Generate an educational report using Qwen2.5-VL through Ollama.
5. Export the results as a PDF or TXT file.

## Model and Evaluation

The classifier checkpoint is stored in `outputs/best_model.pth`. Evaluation information is stored in `outputs/evaluation_report.json`.

For reproducibility, consult `train_model.py` for the training implementation and dataset requirements.

## Limitations and Safety

- The application accepts 2D image files; it does not analyze a complete 3D MRI study.
- Classification scores are not calibrated disease probabilities.
- The generated report may contain errors and requires independent review.
- A qualified medical professional must interpret medical imaging in its clinical context.
- Do not use this experimental application to diagnose patients or guide treatment.

## Author

Developed as an educational AI engineering project.
