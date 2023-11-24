from shiny import run_app
from shiny import App, Inputs, Outputs, Session, reactive, render, run_app, ui
from util.sharedStore import sharedStore

def EDA(x) :

	try:
		headeryn = x.get("headerYN", 'Y')
		sep = x.get("delimiter", ',')
		encode = x.get("fileencoding", 'utf-8')
		dir = x.get("filepathname")
		file = x.get("filename")

		serverInfo = x['serverInfo']
		option = x.get('option', {})

		appDir = serverInfo.get("app_dir", "./src/eda")
		host = serverInfo.get("host", "127.0.0.1")
		port = serverInfo.get("port", 12346)
		
		enable_filedialog = option.get('enable_filedialog', 1)
		
		SHINY_SHAREDATA = {
			"result": "ok",
			"dir": dir,
			"file": file,
			"sep": sep,
			"encode": encode,
			"headeryn": headeryn,
			"enable_filedialog": enable_filedialog
		}
		
		store = sharedStore()
		store.put("SHINY_SHAREDATA", SHINY_SHAREDATA)
		
		run_app(app_dir=appDir, host=host, port=port, autoreload_port=0, reload=False, ws_max_size=16777216, log_level=None, factory=False, launch_browser=False)
		
		retJsonInfo = {
			'data': 'ok',
			'description': ''
		}

	except Exception as e:
		print(e)
		retJsonInfo = {
			'result': 'error',
			'description': str(e)
		}
	
	return retJsonInfo

def main():
	jsonRequest = {
		"method":"EDA",
		"filepathname":"",
		"filename":"",
		"fileencoding":"UTF-8",
		"read_rowcount":300000000000,
		"headerYN":"Y",
		"delimiter":",",
		"serverInfo": {
			"app_dir": "./src/eda",
			"ip": "127.0.0.1",
			"port": 8801
		}
	}
	EDA(jsonRequest)



if __name__ == "__main__":
	main()