from flask import Flask, request, render_template
app = Flask(__name__)

from commons import get_tensor
from inference import get_flower_name
import os

@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index1.html', value='hi')
	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()
		flower_name= get_flower_name(image_bytes=image)
		get_flower_name(image_bytes=image)
		tensor = get_tensor(image_bytes=image)
		#print(get_tensor(image_bytes=image))
		#train_dir = 'dogImages/train'
		#path1=train_dir+'/'+imgclass
		#path2=os.listdir(path1)[0]
		#pathx=imgclass+'/'+path2
		#path3=path1+'/'+path2
		#yy='../'+path3
		#b = f'"{yy}"'
		#y= '<img src='+b+'>'
		return render_template('result.html', flower=flower_name)

if __name__ == '__main__':
	app.run(debug=True)