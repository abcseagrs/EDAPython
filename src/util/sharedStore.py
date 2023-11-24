
globalStore = {}
class sharedStore :
	def __init__(self) :
		global globalStore
		
	def get(self, key):
		if key in globalStore:
			value = globalStore[key]
		else :
			value = None

		return value


	def put(self, key, value):
		globalStore[key] = value


	def petAll(self) :
		return globalStore