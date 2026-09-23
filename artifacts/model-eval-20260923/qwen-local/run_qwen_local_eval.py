"""Real cached Qwen inference through localhost HTTP; no route or mock answers."""
import hashlib, importlib.metadata, json, os, platform, signal, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['LOCAL_POLICY_BASE_URL'] = 'http://127.0.0.1:11503/v1'
os.environ['LOCAL_POLICY_MODEL'] = 'Qwen2.5-1.5B-Instruct'
os.environ['LOCAL_POLICY_MODE'] = 'chat'
os.environ['LOCAL_POLICY_TIMEOUT'] = '90'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import play_local as player

root = Path('output/model-eval-20260923/qwen-local')
root.mkdir(parents=True, exist_ok=True)
snapshot = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
started = time.perf_counter()
manifest = {
 'model':'Qwen/Qwen2.5-1.5B-Instruct','revision':snapshot.name,
 'weights_blob':(snapshot/'model.safetensors').resolve().name,
 'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'device':device,
 'dtype':'float16' if device=='mps' else 'float32','python':platform.python_version(),
 'dependencies':{n:importlib.metadata.version(n) for n in ['torch','transformers','gym-super-mario-bros','gym','nes-py','numpy','httpx','imageio','pillow']},
 'code_sha256':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['play_local.py','mario_env.py','output/run_qwen_local_eval.py']},
 'mode':'model_only','temperature':0,'do_sample':False,'max_new_tokens':8,
 'route':False,'guard':False,'unstick':False,'mentor':False,'rules_fallback':False,
 'level_order':['1-1','1-2','1-3'],'trials':[],
 'notes':'Real cached weights loaded with local_files_only=True. A localhost HTTP adapter forwards unchanged play_local.py messages into the model chat template. No model answer is mocked. One episode per level with no prompt tuning. Emulator pauses during inference. Existing semantic-action executor retained. HTTP trace records JSON bodies, never headers.'}
def save():
 manifest['wall_seconds']=round(time.perf_counter()-started,3)
 (root/'evaluation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
def deadline(_signum,_frame):
 raise TimeoutError('Local evaluation reached 540 second wall-clock budget')
signal.signal(signal.SIGALRM,deadline)
signal.alarm(540)
server = None
current_dir = root
try:
 print('LOAD',device,flush=True)
 torch.set_num_threads(4)
 tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
 model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16 if device=='mps' else torch.float32).to(device).eval()
 manifest['load_seconds']=round(time.perf_counter()-started,3)
 save()
 print('LOADED',manifest['load_seconds'],flush=True)
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):
   pass
  def do_POST(self):
   t=time.perf_counter(); request=None
   try:
    assert self.path == '/v1/chat/completions'
    request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    text=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True)
    inputs=tokenizer([text],return_tensors='pt').to(device)
    with torch.inference_mode():
     output=model.generate(**inputs,max_new_tokens=request['max_tokens'],do_sample=False,temperature=None,top_p=None,top_k=None,max_time=60)
    reply=tokenizer.decode(output[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
    response={'model':'Qwen2.5-1.5B-Instruct','choices':[{'message':{'role':'assistant','content':reply}}],'usage':{'prompt_tokens':int(inputs.input_ids.shape[1]),'completion_tokens':int(output.shape[1]-inputs.input_ids.shape[1])}}
    status=200
   except Exception as exc:
    status=500; response={'error':{'type':type(exc).__name__,'message':str(exc)}}
   latency=round(time.perf_counter()-t,4)
   row={'request':request,'status_code':status,'response':response,'latency_seconds':latency}
   with (current_dir/'http.jsonl').open('a') as f:
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
   body=json.dumps(response).encode()
   self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers()
   try: self.wfile.write(body)
   except BrokenPipeError: pass
   print('INFERENCE',current_dir.name,status,repr(response.get('choices',[{'message':{'content':'error'}}])[0]['message']['content']),latency,flush=True)
 server=HTTPServer(('127.0.0.1',11503),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start()
 for level in manifest['level_order']:
  player.RUNS=root/level; player.RUNS.mkdir(exist_ok=True); current_dir=player.RUNS
  t=time.perf_counter(); print('START',level,flush=True)
  try:
   result=player.run('local',level,model_only=True)
  except Exception as exc:
   result_file=player.RUNS/'results.jsonl'
   result=json.loads(result_file.read_text().splitlines()[-1]) if result_file.exists() else {'level':level,'flag':None}
   result.update(status='error',error_type=type(exc).__name__,error=str(exc))
  result['wall_seconds']=round(time.perf_counter()-t,3)
  manifest['trials'].append(result); save()
  print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
  if result.get('status')=='error':
   manifest['stop_reason']='Stopped after first runtime or invalid-answer error; remaining levels not attempted'; save(); break
except Exception as exc:
 manifest.update(status='error',error_type=type(exc).__name__,error=str(exc)); save(); print('ERROR',type(exc).__name__,str(exc),flush=True)
finally:
 signal.alarm(0)
 if server:
  server.shutdown(); server.server_close()
 save()
print('DONE',flush=True)
