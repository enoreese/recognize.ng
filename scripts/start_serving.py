import os
import signal
import subprocess

# Making sure to use virtual environment libraries
# activate_this = "/home/ubuntu/tensorflow/bin/activate_this.py"
# exec(open(activate_this).read(), dict(__file__=activate_this))

# Change directory to where your Flask's app.py is present
os.chdir("/home/forge/production/recognize.ng/api")
tf_ic_server = ""

try:
    tf_ic_server = subprocess.Popen(["tensorflow_model_server "
                                     "--model_base_path=/home/forge/production/recognize.ng/api/serving/image_quality_classifier "
                                     "--rest_api_port=9000 --model_name=image_quality_classifier"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started TensorFlow Image Quality server!")

    # while True:
    #     print("Type 'exit' and press 'enter' to quit: ")
    #     in_str = input().strip().lower()
    #     if in_str == 'q' or in_str == 'exit':
    #         print('Shutting down all servers...')
    #         os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    #         print('Servers successfully shutdown!')
    #         break
    #     else:
    #         continue
except KeyboardInterrupt:
    print('Shutting down all servers...')
    os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    print('Servers successfully shutdown!')