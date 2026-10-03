import sys
import os
import psutil
import numpy as np
import pandas as pd
import webbrowser
from threading import Timer
import gpu_cpu_v1_3 as gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import threading 
import dash
from dash import Dash, dash_table, html, State, no_update, clientside_callback, dcc, ctx
from dash.dependencies import Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
from pynvml import *
import time
import subprocess
from datetime import datetime

running = True
power_log = []
long_text = ""
currentDIR = os.path.dirname(os.path.abspath(__file__))

#=============================================================================#
#=========================    GPU Check   ====================================#
#=============================================================================#
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#=============================================================================#
#=========================    power monitor   ================================#
#=============================================================================#

def read_gpu_power():
  try:
     result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ],
        text=True
     )
     power_watts, tempC = result.strip().split(", ")
     return float(power_watts), float(tempC)

  except Exception as e:
    return None, None

def CPU_usage():

  #cpu_per_core_usage = psutil.cpu_percent(interval=1, percpu=True)
  #print(cpu_per_core_usage)
  cpu_usage = psutil.cpu_percent(interval=1, percpu=False)

  cpu_freq = psutil.cpu_freq()
  #print(cpu_freq.current, cpu_freq.max)

  coreNum_logical = psutil.cpu_count(logical=True)
  coreNum_physical = psutil.cpu_count(logical=False)

  #print(coreNum_logical, coreNum_physical)

  cpu_mem_usage = psutil.virtual_memory()
  #print(cpu_mem_usage.percent)

  return float(cpu_usage), float(cpu_freq.current), int(coreNum_logical), int(coreNum_physical), float(cpu_mem_usage.percent)

#=============================================================================#
#=============================   dash: UI   ==================================#
#=============================================================================#

app = Dash(__name__)


#============================= top panel =====================================#
app.layout = html.Div([
   html.H2(
     "Heterogeneous Computation Data Viewer",
     style={
        "textAlign": "center",
        "fontSize": "40px"}),
        
   html.Br(),

   html.Div([
     html.Button("Quit",
       id="btn-quit",
       n_clicks=0,
       style={
         "width": "100px",
         "height": "50px",
         "fontSize": "20px",
         "justifyContent": "right"}
     )
   ],
   style = {
     "display": "flex",
     "justifyContent": "flex-start",
     "gap": "100px"}
   ),

   html.Div(id="quit-status"),

   html.Br(),
   html.Div([
     html.H3("Monitor GPU/CPU",
       style={
         "textAlign": "left",
         "fontSize": "20px"}),
    
     dcc.Interval(
       id="monitor-timer",
       interval=2000,      # update every 1000 ms = 1 sec
       n_intervals=0,
       disabled=True
     ),
   dcc.Store(id="gpu-log", data=""),
   dcc.Store(id="cpu-log", data=""),
   html.Div([
      html.Pre(
        id="gpu-power-box",
        children="",
        style={
            "height": "100px",
            "width": "400px",
            "overflowY": "scroll",
            "whiteSpace": "pre-wrap",
            "border": "1px solid black",
            "padding": "10px"
        }
      ),
      html.Pre(
        id="cpu-power-box",
        children="",
        style={
            "height": "100px",
            "width": "400px",
            "overflowY": "scroll",
            "whiteSpace": "pre-wrap",
            "border": "1px solid black",
            "padding": "10px"
        }
      )
   ],
   style={"display": "flex"}
   ),
   html.Div(id="dummy-scroll", style={"display": "none"}),
   
   html.Div([
      html.Br(),
      html.Button("Start", 
                 id="btn-start",
                 style={
                    "width": "100px",
                    "height": "50px",
                    "fontSize": "20px",
                    "justifyContent": "right"}),
      html.Button("Stop", 
                id="btn-stop",
                style={
                    "width": "100px",
                    "height": "50px",
                    "fontSize": "20px",
                    "justifyContent": "right"}
      )
   ])
   ],
   
   style = {
     "display": "flex",
     "gap": "20px"}
   ),
   html.Br(),

   #======================== END: top panel ============================================#
   #------------------------------------------------------------------------------------#
   #======================== START: GPU/CPU benchmark tab ==============================#
   dcc.Tabs([
     dcc.Tab(
       label="GPU/CPU Runtime Benchmark", 
       style = {"fontSize": "25px"},
       children=[
         html.Br(),
         html.Div(
           dcc.Checklist(
             id = "test-checklist",
             options =[
                {"label": "Vector Addition", "value": "VA"},
                {"label": "Dot Product", "value": "DP"}],
             style = {"display": "flex",
                        "gap": "10px",
                        "fontSize": "20px"}
           )
         ),
         html.Div(id = "test_selection"),
         html.Br(),
         html.Div([
           html.Button("Start", 
             id="btn-performance", 
             n_clicks=0, 
             style={
               "width": "100px", 
               "height": "50px", 
               "fontSize": "20px"
             }
           )
         ]),
     
       html.Br(),
         html.Div([
           dcc.Graph(id="performance-GPU_nw", style={"width": "600px", "height": "400px"}), 
           dcc.Graph(id="matrix-GPU_nw", style={"width": "600px", "height": "400px"})
         ], 
         style = {"display": "flex", "justifyContent": "center", "gap": "5px"}
       ), 

       html.Div([
          dcc.Graph(id="performance-GPU_w", style={"width": "600px", "height": "400px"}), 
          dcc.Graph(id="matrix-GPU_w", style={"width": "600px", "height": "400px"})
       ],
       style = {"display": "flex", "justifyContent": "center", "gap": "5px"}
       ), 

       html.Div([
          dcc.Graph(id="performance-CPU1", style={"width": "600px", "height": "400px"}), 
          dcc.Graph(id="matrix-CPU1", style={"width": "600px", "height": "400px"})
          ],
          style = {"display": "flex", "justifyContent": "center", "gap": "5px"}
        ), 

       html.Div([
          dcc.Graph(id="performance-CPU24", style={"width": "600px", "height": "400px"}), 
          dcc.Graph(id="matrix-CPU24", style={"width": "600px", "height": "400px"})
          ],
          style = {"display": "flex", "justifyContent": "center", "gap": "5px"}
       )]
     ),
      #======================== END: GPU/CPU benchmark tab ==============================#
      #----------------------------------------------------------------------------------#
      #======================== START: noise cancel tab =================================#
      dcc.Tab(
        label="Noise Cancellation",
        style = {"fontSize": "25px"},
        children=[
          html.Br(),
          html.Div([
            html.Div([
              dcc.RadioItems(
                id="radio-noiseCancel",
                options=[
                  {"label": "CPU", "value": "cpu"},
                  {"label": "GPU: AI(DeepFilterNet)", "value": "gpu"},
                  {"label": "FPGA", "value": "fpga"}
                ],
                style={
                  "fontSize": "20px",
                  "gap": "10px"},
                value="gpu"      # default selection
              )
            ]),
            html.Div(id="output-noiseCancel"),

            html.Div([
              dcc.Upload(
                id="upload-noisyAudio",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select Noisy WAV File")
                ]),
                style = {
                  'width': '350px',
                  'height': '70px',
                  'lineHeight': '60px',
                  'borderWidth': '2px',
                  'borderStyle': 'dashed',
                  'borderColor': 'black',
                  #'borderRadius': '5px',
                  'textAlign': 'center',
                  'margin': '10px',
                  "fontSize": "20px"},
                multiple = False
              )],
              style={"display": "flex", "justifyContent": "flex-start"}
            ),
            
            html.Div(id="output-noisyAudio-upload"),
            html.Div([
              html.Button(
                "Process", 
                id="btn-noiseCancel", 
                n_clicks=0, 
                style={
                  "width": "120px", 
                  "height": "50px",
                  "fontSize": "20px",
                  'margin': '10px'
                }
              )
            ])
          ]),
          html.Div(id="output-cleanAudio-upload")
        ]
      ),

      #======================== END: noise cancel tab ===================================#
      #----------------------------------------------------------------------------------#
      #======================== START: FIR filter tab ===================================#
      dcc.Tab(
        label="FIR Filter Implementation",
        style = {"fontSize": "25px"},
        children=[
          html.Br(),
          html.Div(
            [
              dcc.RadioItems(
                id="radio-test",
                options=[
                  {"label": "CPU: dll(C/C++)", "value": "dll"},
                  {"label": "GPU: Python", "value": "py"},
                  {"label": "FPGA: Verilog", "value": "fpga"}
                ], 
                value="py"      # default selection
              )
            ]
          ),
          html.Div(id="output-fir"),
          html.Br(),
          html.Div(
            [
              html.Button(
                "Run",
                id="btn-firRun",
                n_clicks=0,
                style={
                  "width": "120px", 
                  "height": "50px",
                  "fontSize": "20px",
                  'margin': '10px'
                }
              )
            ],
            style={
              "display":"flex",
              "gap":"10px"}
          ),
          html.Div(id="output-firRun"),

          html.Br(),
          html.Div(
            [
              dcc.Graph(id="fir-pre", style={"width": "600px", "height": "400px"}), 
              dcc.Graph(id="fir-post", style={"width": "600px", "height": "400px"})
            ], 
            style = {"display": "flex", "justifyContent": "center", "gap": "5px"}
          ), 
        ]
      ),
     
    ])
])
      #======================== END: FIR filter tab =====================================#

#=============================================================================#
#=======================   dash: callback   ==================================#
#=============================================================================#
@app.callback(
    Output("gpu-log", "data"),
    Output("cpu-log", "data"),
    Output("gpu-power-box", "children"),
    Output("cpu-power-box", "children"),
    Input("monitor-timer", "n_intervals"),
    State("gpu-log", "data"),
    State("cpu-log", "data")
)
def update_monitors(n, gpu_log, cpu_log):

    gpu_log = gpu_log or ""
    cpu_log = cpu_log or ""

    now = datetime.now().strftime("%H:%M:%S")

    # Read values
    power, tempC = read_gpu_power()
    cpu_usage, cpu_freq, logical_cores, physical_cores, mem_usage = CPU_usage()

    # GPU line
    if power is None or tempC is None:
        gpu_line = f"{now} | GPU read failed\n"
    else:
        gpu_line = f"{now} |[GPU Power={power:.2f}W], [GPU Temp={tempC}C]\n"

    # CPU line
    if isinstance(cpu_usage, list):
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
    else:
        avg_cpu = cpu_usage

    cpu_line = (
        f"{now} | [CPU Usage={avg_cpu:.2f}%], "
       #f"[Freq={cpu_freq:.2f} MHz], "
        #f"[Core #: logical={logical_cores}, physical={physical_cores}], "
        #f"Physical = {physical_cores}, "
        f"[Memory={mem_usage:.2f}%]\n"
    )

    # Append logs
    gpu_log += gpu_line
    cpu_log += cpu_line

    return (
        gpu_log,
        cpu_log,
        gpu_log,
        cpu_log
    )
    


@app.callback(
    Output("monitor-timer", "disabled"),
    Input("btn-start", "n_clicks"),
    Input("btn-stop", "n_clicks"),
    prevent_initial_call=True
)
def control_monitor(start_clicks, stop_clicks):

    ctx = dash.callback_context

    if not ctx.triggered:
        return True

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-start":
        return False   # start timer

    if button_id == "btn-stop":
        return True    # stop timer

    return True


clientside_callback(
    """
    function(gpu_children, cpu_children) {

        const gpu_box = document.getElementById("gpu-power-box");
        const cpu_box = document.getElementById("cpu-power-box");

        if (gpu_box) {
            gpu_box.scrollTop = gpu_box.scrollHeight;
        }

        if (cpu_box) {
            cpu_box.scrollTop = cpu_box.scrollHeight;
        }

        return "";
    }
    """,
    Output("dummy-scroll", "children"),
    Input("gpu-power-box", "children"),
    Input("cpu-power-box", "children")
)



@app.callback(  
    Output("performance-GPU_nw", "figure"),
    Output("performance-GPU_w", "figure"),
    Output("performance-CPU1", "figure"),
    Output("performance-CPU24", "figure"),
    Output("matrix-GPU_nw", "figure"),
    Output("matrix-GPU_w", "figure"),
    Output("matrix-CPU1", "figure"),
    Output("matrix-CPU24", "figure"),
    Input("btn-performance", "n_clicks"),
    State("test-checklist", "value"))


def update_performance_graph(n_clicks,test_selected):
    scaled_arrSize =[] 
    cpuFloatArr1 = []
    cpuFloatArr24 = []
    gpuFloatArr_w = [] 
    gpuFloatArr_nw = []
    arrSize = []
    gpuArr_nw = []
    gpuArr_w = []
    cpuArr1 = []
    cpuArr24 = []
    y_max_va = 1
    y_max_dp = 1
    
    VA_GPU_nw = go.Figure()
    VA_GPU_nw.update_layout(title="Vector Addition GPU w/o WarmUp")
    VA_GPU_w = go.Figure()
    VA_GPU_w.update_layout(title="Vector Addition GPU w/ WarmUp")
    VA_CPU1 = go.Figure()
    VA_CPU1.update_layout(title="Vector Addition CPU 1-thread")
    VA_CPU24 = go.Figure()
    VA_CPU24.update_layout(title="Vector Addition CPU 24-threads")
    DP_GPU_nw = go.Figure()
    DP_GPU_nw.update_layout(title="Dot Product GPU w/o WarmUp")
    DP_GPU_w = go.Figure()
    DP_GPU_w.update_layout(title="Dot Product GPU w/ WarmUp")
    DP_CPU1 = go.Figure()
    DP_CPU1.update_layout(title="Dot Product CPU 1-thread")
    DP_CPU24 = go.Figure()
    DP_CPU24.update_layout(title="Dot Product CPU 24-threads")


    if n_clicks == 0:
      return VA_GPU_nw, VA_GPU_w, VA_CPU1, VA_CPU24, DP_GPU_nw, DP_GPU_w, DP_CPU1, DP_CPU24

    if test_selected is None:
      scaled_arrSize =[] 
      cpuFloatArr1 = []
      cpuFloatArr24 = []
      gpuFloatArr_w = [] 
      gpuFloatArr_nw = []
      arrSize = []
      gpuArr_nw = []
      gpuArr_w = []
      cpuArr1 = []
      cpuArr24 = []

    if "VA" in test_selected:
      scaled_arrSize, cpuFloatArr1, cpuFloatArr24, gpuFloatArr_w, gpuFloatArr_nw = gc.performance_benchmark()
      y_max_va = max(np.max(cpuFloatArr1), np.max(cpuFloatArr24), np.max(gpuFloatArr_w), np.max(gpuFloatArr_nw)) 
    

    if "DP" in test_selected:
      arrSize, gpuArr_nw, gpuArr_w, cpuArr1, cpuArr24 = gc.matrix_multiplication_benchmark()
      y_max_dp = max(np.max(gpuArr_nw), np.max(gpuArr_w), np.max(cpuArr1), np.max(cpuArr24)) 

   
    title = "Vector Addition GPU w/o WarmUp"
    VA_GPU_nw = go.Figure()
    VA_GPU_nw.add_trace(go.Scatter(x=scaled_arrSize, y=gpuFloatArr_nw, mode="lines", name=title))
    VA_GPU_nw.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_va]))

    title = "Vector Addition GPU w/ WarmUp"
    VA_GPU_w = go.Figure()
    VA_GPU_w.add_trace(go.Scatter(x=scaled_arrSize, y=gpuFloatArr_w, mode="lines", name=title))
    VA_GPU_w.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_va]))

    title = "Vector Addition CPU 1-thread"
    VA_CPU1 = go.Figure()
    VA_CPU1.add_trace(go.Scatter(x=scaled_arrSize, y=cpuFloatArr1, mode="lines", name=title))
    VA_CPU1.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_va]))

    title = "Vector Addition CPU 24-threads"
    VA_CPU24 = go.Figure()
    VA_CPU24.add_trace(go.Scatter(x=scaled_arrSize, y=cpuFloatArr24, mode="lines", name=title))
    VA_CPU24.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_va]))
    
    title = "Dot Product GPU w/o WarmUp"
    DP_GPU_nw = go.Figure()
    DP_GPU_nw.add_trace(go.Scatter(x=arrSize, y=gpuArr_nw, mode="lines", name=title))
    DP_GPU_nw.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_dp]))

    title = "Dot Product GPU w WarmUp"
    DP_GPU_w = go.Figure()
    DP_GPU_w.add_trace(go.Scatter(x=arrSize, y=gpuArr_w, mode="lines", name=title))
    DP_GPU_w.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_dp]))

    title = "Dot Product CPU 1-thread"
    DP_CPU1 = go.Figure()
    DP_CPU1.add_trace(go.Scatter(x=arrSize, y=cpuArr1, mode="lines", name=title))
    DP_CPU1.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_dp]))

    title = "Dot Product CPU 24-threads"
    DP_CPU24 = go.Figure()
    DP_CPU24.add_trace(go.Scatter(x=arrSize, y=cpuArr24, mode="lines", name=title))
    DP_CPU24.update_layout(title=title, xaxis_title="array size", yaxis_title="test time(s)", yaxis=dict(
          range=[0, y_max_dp]))

    return VA_GPU_nw, VA_GPU_w, VA_CPU1, VA_CPU24, DP_GPU_nw, DP_GPU_w, DP_CPU1, DP_CPU24

@app.callback(
    Output("output-cleanAudio-upload", "children"),
    Input("btn-noiseCancel", "n_clicks"),
    State("radio-noiseCancel", "value"),
    State("upload-noisyAudio", "contents"),
    State("upload-noisyAudio", "filename"),
    prevent_initial_call=True
)

def update_output(n_clicks, target_noiseCancel, contents, filename):
    import base64
    import noise_cancel as nc
    import time
    from dash import html
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
    if not contents or not filename:
        return "Please upload a noisy audio file first."

    assets_dir = os.path.join(currentDIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    input_path = os.path.join(assets_dir, "noisy.wav")
    output_path = os.path.join(assets_dir, "cleaned.wav")

    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    with open(input_path, "wb") as f:
        f.write(decoded)
    
    match target_noiseCancel:
      case "gpu": 
        nc.preTrained_deepFilterNet(input_path, output_path) 
        noisyPath = f"/assets/noisy.wav?v={time.time()}"
        cleanPath = f"/assets/cleaned.wav?v={time.time()}"
      
      case "cpu":
        noisyPath = ""
        cleanPath = ""

      case "fpga":
        noisyPath = ""
        cleanPath = ""

    print("cleaned exists:", os.path.exists(output_path))
    print("cleaned size:", os.path.getsize(output_path))

    return html.Div(
      [
        html.Br(),
        html.Div(
          [
             html.Span(
                "Noisy Audio",
                style={"fontWeight": "bold", "marginRight": "10px"},
            ),
            html.Audio(
                title="noisy audio",
                #src=f"/assets/noisy.wav?v={time.time()}",
                src=noisyPath,
                controls=True
            ),
            html.Span(
                "Cleaned Audio",
                style={"fontWeight": "bold", "marginRight": "10px"},
            ),
            html.Audio(
                title="cleaned audio",
                #src=f"/assets/cleaned.wav?v={time.time()}",
                src=cleanPath,
                controls=True
            )
          ],
          style = {"display": "flex"}
        )
      ]
    )

@app.callback(
    Output("fir-pre", "figure"),
    Output("fir-post", "figure"),
    Input("btn-firRun", "n_clicks"),
    State("radio-test", "value"))

def update_output(n_clicks, selected):
  import ctypes
  import FIR as fPy
  import cupy as cp

  DLL_FILE = f'{currentDIR}/fir_dll/fir_dll.dll'
  
  preData = []
  arrSize = []
  y_max_pre = 1
  y_max_post = 1
  y_min_pre = -1
  y_min_post = -1
  postData = []
  x_arr_pre = []
  x_arr_post = []
  pre_FIR = go.Figure()
  pre_FIR.update_layout(title="Pre-FIR filter Data")
  post_FIR = go.Figure()
  post_FIR.update_layout(title="Post-FIR filter Data")

  if n_clicks == 0:
      return pre_FIR, post_FIR

  match selected:
    case "dll":
      import fir_dll as fd

      fs = 100
      duration = 5
      num_taps = 32
      x_size = fs * duration
      x_arr_pre = np.arange(x_size) / fs
      preData = (
              np.sin(2 * np.pi * 1 * x_arr_pre)
              + 0.3 * np.sin(2 * np.pi * 8 * x_arr_pre)
          ).astype(np.float32)
      postData = np.zeros(x_size+num_taps-1, dtype=np.float32)
      coeffTaps = np.ones(num_taps, dtype=np.float32) / num_taps
      x_arr_post = np.arange(x_size + num_taps - 1) / fs
      fd.import_DLL(DLL_FILE, preData, coeffTaps, postData, x_size, num_taps)
      y_max_pre = np.max(preData)
      y_min_pre = np.min(preData)
      y_max_post = np.max(postData)
      y_min_post = np.min(postData)

    case "py":
      fs = 100
      duration = 5.0
      num_taps = 32
      x_size = fs * duration
      x_arr_pre = np.arange(x_size) / fs
      x_arr_post = np.arange(x_size + num_taps - 1) / fs
      t = np.arange(0, duration, 1 / fs)
      h = np.ones(num_taps) / num_taps

      preData = np.sin(2*np.pi*1*t) + 0.3*np.sin(2*np.pi*8*t)
      postData = fPy.fir_filter_gpu(preData, h)
      y_max_pre = np.max(preData)
      y_min_pre = np.min(preData)
      y_max_post = np.max(postData)
      y_min_post = np.min(postData)

    case "fpga":





      print("fpga")

  title="Pre-FIR filter Data"
  pre_FIR = go.Figure()
  pre_FIR.add_trace(go.Scatter(x=x_arr_pre, y=preData, mode="lines", name=title))
  pre_FIR.update_layout(title=title, xaxis_title="frequency", yaxis_title="amplitude", yaxis=dict(
    range=[y_min_pre, y_max_pre]))

  title="Post-FIR filter Data"
  post_FIR = go.Figure()
  post_FIR.add_trace(go.Scatter(x=x_arr_post, y=postData, mode="lines", name=title))
  post_FIR.update_layout(title=title, xaxis_title="frequency", yaxis_title="amplitude", yaxis=dict(
    range=[y_min_post, y_max_post]))

  return pre_FIR, post_FIR



@app.callback(
    Output("quit-status", "children"),
    Input("btn-quit", "n_clicks"), 
    prevent_initial_call=True
)
def quit_app(n_clicks):
    os._exit(0)
    return ""



#=================================================================================================#
#================================    main    =====================================================#
#=================================================================================================#
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=False)
    
