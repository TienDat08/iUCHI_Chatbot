**Prepare**
- create venv: 
```
python3 -m venv myenv
```
- install dependencies: 
```
pip install -r requirements.txt
```
- place api keys to .env file, require *GEMINI_API_KEY* and *TAVILY_API_KEY*

**Run**
- to test workflow and see each components/agents:
```terminal
python single_test.py
```
- to test the agents in Fastapi app with unvicorn server, [implement](/agents/app.py):
```terminal
python api_test.py
```
go to http://127.0.0.1:8000/docs to try the swagger