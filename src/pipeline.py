#goal : index all the images using llamaindex and save it to qdrant 
from llama_cloud import LlamaCloud;
from designsmith.utils.file_utility import download_files_from_s3, list_objects;

#need to configure the llama cloud first
#using async configurations 
file_vector = list_objects();
client = LlamaCloud()

#function is the main indexing function
def indexer():
    file = client.files.create(file="", purpose="")
    result = client.parsing.parse(file_id)