from langserve import RemoteRunnable


remote_chain = RemoteRunnable("http://localhost:8000/chain/")

remote_chain.invoke({"language": "italian", "text": "hi"})
