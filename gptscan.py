import tkinter as tk
import os
import tkinter.ttk as ttk
import tensorflow as tf
import json
from functools import partial
import tkinter.font
import openai

MAXLEN=1024

apikey=''
try:
    with open('apikey.txt', 'r') as file:
        apikey = file.readline().strip()
except FileNotFoundError:
    print("OpenAI key file not found. No GPT data will be included in report...")

taskdesc=''
try:
    with open('task.txt', 'r') as file:
        taskdesc = file.readline().strip()
except FileNotFoundError:
    print("Task description file not found. No GPT data will be included in report...")
if taskdesc=='':
    apikey='' #if task description missing, null the API key to not waste tokens

try:
    with open('extensions.txt', 'r') as file:
        lines = file.readlines()
        extensions = [line.strip() for line in lines]
except FileNotFoundError:
    print("Extensions list not found! Will not be able to scan, exiting.")
    exit()

def motion_handler(tree, event):
    f = tkinter.font.Font(font='TkDefaultFont')
    # A helper function that will wrap a given value based on column width
    def adjust_newlines(val, width, pad=10):
        if not isinstance(val, str):
            return val
        else:
            words = val.split()
            lines = [[],]
            for word in words:
                line = lines[-1] + [word,]
                if f.measure(' '.join(line)) < (width - pad):
                    lines[-1].append(word)
                else:
                    lines[-1] = ' '.join(lines[-1])
                    lines.append([word,])

            if isinstance(lines[-1], list):
                lines[-1] = ' '.join(lines[-1])

            return '\n'.join(lines)

    if (event is None) or (tree.identify_region(event.x, event.y) == "separator"):
        # You may be able to use this to only adjust the two columns that you care about
        # print(tree.identify_column(event.x))

        col_widths = [tree.column(cid)['width'] for cid in tree['columns']]

        for iid in tree.get_children():
            new_vals = []
            for (v,w) in zip(tree.item(iid)['values'], col_widths):
                new_vals.append(adjust_newlines(v, w))
            tree.item(iid, values=new_vals)

def list_files(path):
    file_list = []
    for root, directories, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_list.append(file_path)
    return file_list

def sort_column(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children("")]
    if col == "own_conf" or col =="gpt_conf":
        # Convert percentage strings to floats for sorting
        l = [(float(val.strip("%")), k) for val, k in l]
    else:
        # Treat other columns as text
        l = [(val, k) for val, k in l]

    l.sort(reverse=reverse)

    for index, (val, k) in enumerate(l):
        tv.move(k, "", index)

    # Reverse sort order on subsequent clicks of the same column header
    tv.heading(col, command=lambda: sort_column(tv, col, not reverse))

    
def button_click():
    modelscript = tf.keras.models.load_model('scripts.h5')
    file_list=list_files(textbox.get())
    for file_path in file_list:
        extension = os.path.splitext(file_path)[1]
        if any(extension == ext.lower() for ext in extensions):
            print(file_path)
            with open(file_path, 'rb') as f:
                data = list(f.read())
            resultchecks=[]
            if len(data)<=MAXLEN:
                numtoadd=MAXLEN-len(data)
                data.extend([13]*numtoadd)
            file_size=max(MAXLEN,len(data))
            tf_data = tf.expand_dims(tf.constant(data), axis=0)

            maxconf_pos=0
            maxconf=0
            for i in range(0, file_size-MAXLEN+1, MAXLEN):
                #end is file_size-MAXLEN because last bytes will be later scanned in a full buffer
                if i>=MAXLEN and deep_var.get()==False:
                    continue
                print ("Scanning at:", i)
                result=modelscript.predict(tf_data[:, i:i+1024], batch_size=1, steps=1)[0][0]
                resultchecks.append(result)
                if result>maxconf:
                    maxconf_pos=i
                    maxconf=result


            #if greater than MAXLEN, always scan last MAXLEN, even if some bytes get scanned twice.
            #getting last full MAXLEN into buffer is important because of appending viruses
            if file_size>MAXLEN:
                print ("Scanning at:", -MAXLEN)
                result=modelscript.predict(tf_data[:, -MAXLEN:], batch_size=1, steps=1)[0][0]
                resultchecks.append(result)
                if result>maxconf:
                    maxconf_pos=file_size-MAXLEN
                    maxconf=result
            percent = "{:.0%}".format(maxconf)
            admin_desc=""
            enduser_desc=""
            chatgpt_conf=0
            snippet=''.join(map(chr,bytes(data[maxconf_pos:maxconf_pos+1024]))).strip()
            if max(resultchecks)>.5 and gpt_var.get()==True:
                openai.api_key=apikey
                try:
                    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=[
                            {"role": "system", "content": taskdesc},
                            {"role": "user", "content": snippet},
                        ]
                      )
                    print(response)
                    json_data = json.loads(response.choices[0].message.content)
                    admin_desc=json_data["administrator"]
                    enduser_desc=json_data["end-user"]
                    try:
                        chatgpt_conf = int(json_data["threat-level"])
                    except ValueError:
                        chatgpt_conf=-1
                    chatgpt_conf=json_data["threat-level"]
                except json.JSONDecodeError as e:
                    admin_desc='JSON Parse Error'
                    enduser_desc='JSON Parse Error'
                    chatgpt_conf=0
                if chatgpt_conf==-1:
                    admin_desc='JSON Parse Error'
                    enduser_desc='JSON Parse Error'
            snippet=''.join([s for s in snippet.strip().splitlines(True) if s.strip()])
            chatgpt_conf_percent="{:.0%}".format(chatgpt_conf/100.)
            if max(resultchecks)>.5 or all_var.get()==True:
                tree.insert("", tk.END, values=(file_path,percent,admin_desc,enduser_desc,
                                                chatgpt_conf_percent,snippet))

root = tk.Tk()
root.geometry("800x500")
root.title("GPT Virus Scanner")

label = tk.Label(root, text="Path to scan")
label.pack()

textbox = tk.Entry(root)
textbox.pack()

deep_var = tk.BooleanVar()
deep_checkbox = tk.Checkbutton(root, text="Deep scan", variable=deep_var)
deep_checkbox.pack()

all_var = tk.BooleanVar()
all_checkbox = tk.Checkbutton(root, text="Show all files", variable=all_var)
all_checkbox.pack()

gpt_var = tk.BooleanVar()
gpt_checkbox = tk.Checkbutton(root, text="Use ChatGPT", variable=gpt_var)
gpt_checkbox.pack()

button = tk.Button(root, text="Scan now", command=button_click)
button.pack()
style = ttk.Style(root)
style.configure('Scanner.Treeview', rowheight=100)
tree = ttk.Treeview(root, style='Scanner.Treeview')
tree["columns"] = ("path", "own_conf", "admin_desc", "end-user_desc", "gpt_conf", "snippet")
tree.column("#0", width=0, stretch=tk.NO)
tree.column("path", width=200, stretch=tk.NO, anchor="w")
tree.column("own_conf", width=50, stretch=tk.NO, anchor="e")
tree.column("admin_desc", width=200, stretch=tk.NO, anchor="w")
tree.column("end-user_desc", width=200, stretch=tk.NO, anchor="w")
tree.column("gpt_conf", width=50, stretch=tk.NO, anchor="e")
tree.column("snippet", width=50, stretch=tk.NO, anchor="w")

tree.heading("#0", text="")
tree.heading("path", text="File Path", command=lambda: sort_column(tree, "path", False))
tree.heading("own_conf", text="Own confidence",
             command=lambda: sort_column(tree, "own_conf", False))
tree.heading("admin_desc", text="Administrator Description",
             command=lambda: sort_column(tree, "admin_desc", False))
tree.heading("end-user_desc", text="End-User Description",
             command=lambda: sort_column(tree, "end-user_desc", False))
tree.heading("gpt_conf", text="ChatGPT confidence",
             command=lambda: sort_column(tree, "gpt_conf", False))
tree.heading("snippet", text="Snippet", command=lambda: sort_column(tree, "snippet", False))

tree.pack(fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

tree.configure(yscrollcommand=scrollbar.set)
tree.bind('<B1-Motion>', partial(motion_handler, tree))
motion_handler(tree, None)   # Perform initial wrapping

root.mainloop()


