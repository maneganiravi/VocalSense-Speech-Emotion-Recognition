# VocalSense: Speech Emotion Recognition Using CNN and CNN-BiLSTM Ensemble Learning

VocalSense is a local web application for speech emotion recognition. It combines predictions from a CNN and a CNN-BiLSTM by averaging their class probabilities. It does not invent results: the prediction API stays unavailable until both models have been trained.

The RAVDESS dataset configured for this workspace was verified at:

`C:\Users\M.Ravi kumar\OneDrive - Shri Vile Parle Kelavani Mandal\RAVDESS`

Its top-level `Actor_01`–`Actor_24` folders contain 1,440 speech WAV files with valid filename emotion codes for all eight classes. A nested `audio_speech_actors_01-24` folder is also present; training reads the top-level actor folders only. The generated local `dataset_config.json` is ignored by Git; change its `dataset_path` value if you move the dataset.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.11 (recommended for TensorFlow on Windows)
- The RAVDESS speech audio dataset

## First test: start the untrained application

First check that the Python 3.11 launcher is available:

```powershell
py -3.11 --version
```

If it is not installed, install Python 3.11 and reopen PowerShell:

```powershell
winget install --id Python.Python.3.11 --exact
```

Open PowerShell in the `VocalSense` project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

If PowerShell prevents environment activation, run this once in the same window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open **http://127.0.0.1:8000** in your browser. The page and dataset status should load. Until training is complete, the app reports that models are not trained; prediction is correctly disabled. This is expected for the first test.

You can also check the API directly at **http://127.0.0.1:8000/docs** or test health in another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/samples
```

## Train the models

Stop the web server with `Ctrl+C`. Keep the virtual environment activated and make sure `dataset_config.json` points to the folder containing `Actor_01`, `Actor_02`, and the other actor folders. From the project folder:

```powershell
python -m training.train
```

Training checks RAVDESS filenames and class coverage, skips unreadable audio with a warning, and splits whole actors between training and held-out testing (seed 42). It trains both models with shared 22,050 Hz mono log-mel preprocessing and saves the models and actual held-out metrics in `models\`.

The first training run can take a while, especially on a CPU. TensorFlow may use available hardware. Do not close PowerShell until training finishes.

## Evaluate the saved models

```powershell
python -m training.evaluate
```

The evaluation prints accuracy, macro and weighted F1, macro precision and recall, and the confusion matrix for each model on the held-out actors. These values are calculated from your actual dataset and trained model files.

## Start and use VocalSense

```powershell
python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000**, choose a WAV file, listen to it with the audio player, and select **Analyze emotion**. Supported uploads must be WAV, no larger than 15 MB, and between 0.5 and 30 seconds. Prediction history is stored in `database\vocalsense.db`.

The interface also has model performance, prediction history, feature information, and settings/about pages. Model scores are shown only after actual training metrics have been saved.

## Dataset path

Update `dataset_config.json` if your RAVDESS folder moves. The path should point to the directory directly containing the `Actor_*` folders, for example:

```json
{
  "dataset_path": "D:\\datasets\\RAVDESS"
}
```

The recognized filename structure is `03-01-EE-...-AA.wav`: `03` selects speech audio, `EE` is the emotion code, and `AA` identifies the actor. Files with invalid names or actor-folder mismatches are excluded. Emotion codes `01`–`08` mean Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## API endpoints

- `GET /api/health` — API and trained-model readiness.
- `GET /api/samples` — verified RAVDESS file and actor counts, emotion counts, and example filenames.
- `GET /api/dataset-status` — configured dataset path, directory availability, discovered WAV count, valid clip count, actor/emotion counts, and diagnostics.
- `GET /api/dashboard` — dashboard summary including dataset status, loaded model count, and prediction total.
- `GET /api/sample-audio?filename=...` — serves a selected sample WAV from the configured dataset.
- `POST /api/predict` — multipart WAV upload; returns each model's prediction and the probability-averaged ensemble.
- `GET /api/history` — latest locally saved predictions.
- `GET /api/model-info` — preprocessing settings and real held-out evaluation metrics.
