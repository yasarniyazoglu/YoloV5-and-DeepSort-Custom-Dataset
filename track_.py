import threading

# 여러 스레드에서 함수 호출
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import shutil
import platform
import os
import argparse
import subprocess
from datetime import datetime
import time, json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import boto3
from botocore.client import Config
import map
from dotenv import load_dotenv
import threading
lock = threading.Lock()

load_dotenv()
HLS_PATH = os.environ.get('HLSPATH')
HLS_OUTPUT = HLS_PATH

# 영상 정보
video_width = 720
video_height = 480
fps = 6

# S3 영상 저장
ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID')
ACCESS_SECRET_KEY = os.environ.get('ACCESS_SECRET_KEY')
BUCKET_NAME = 'traffic-inf'
s3 = ""
savePeriod = 30

# IoT
CLIENT_ID = "MyTest"
ENDPOINT =  os.environ.get('ENDPOINT')
PATH_TO_AMAZON_ROOT_CA_1 =  os.environ.get('PATH_TO_AMAZON_ROOT_CA_1')
PATH_TO_PRIVATE_KEY =  os.environ.get('PATH_TO_PRIVATE_KEY')
PATH_TO_CERTIFICATE = os.environ.get('PATH_TO_CERTIFICATE')
MESSAGE = "test"
TOPIC = "cnt" 
RANGE = 20
myMQTTClinet = ""


# input video
videoType = ['in', 'out', 'center']
videoTypeNum = 0

# 인원수 카운팅
busNum = os.environ.get('BUSNUM')
incount = 0
outcount = 0
priorcount = -1
inCountIds = []
outCountIds = []
line = []  # x1, y1, x2, y2

# fall detection
fallIdx = 1
fallIds = []
transmit = False
transmitFrame = 0

def IoTInit():
    global myMQTTClinet
    myMQTTClinet = AWSIoTMQTTClient(CLIENT_ID)
    myMQTTClinet.configureEndpoint(ENDPOINT, 8883)
    myMQTTClinet.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
    myMQTTClinet.configureOfflinePublishQueueing(-1)
    myMQTTClinet.configureDrainingFrequency(2)
    myMQTTClinet.configureConnectDisconnectTimeout(60)
    myMQTTClinet.configureMQTTOperationTimeout(30)
    print("Initiating IoT Core Topic ...")
    myMQTTClinet.connect()

def S3Init():
    global s3
    s3 = boto3.resource(
            's3',
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=ACCESS_SECRET_KEY,
            config=Config(signature_version='s3v4')
        )
    
def trackInit(opt):
    # Initialize
    device = select_device(opt.device)
    out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
    opt.output, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
    opt.save_txt, opt.img_size, opt.evaluate
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(
    model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    return out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, device, model, stride, names, vid_path, half

def handle_upload_img(file): # f = 파일명 이름.확장자 분리
    print("upload_img!!")
    if "ts" in file:
        typ = "video/MP2T"
    else:
        typ = "application/x-mpegURL"

    data = open(HLS_OUTPUT + file, 'rb')
    # '로컬의 해당파일경로'+ 파일명 + 확장자
    s3.Bucket(BUCKET_NAME).put_object(
        Key= f'{busNum}/{file}', Body=data, ContentType=typ)

def run_ffmpeg(width, height, fps):
    ffmpg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',  # x264 비디오 코덱 지정
        '-g', f'{fps*10}',  # 키프레임 간격 설정 (여기서는 10으로 예시로 설정)  
        '-hls_time', '10',
        '-hls_list_size', '10',
        '-force_key_frames', f'expr:gte(t,n_forced*{fps*10})',  # 키프레임 간격을 맞추기 위한 설정
        f'{HLS_OUTPUT}index.m3u8'
    ]
    return subprocess.Popen(ffmpg_cmd, stdin=subprocess.PIPE)

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def isFall(track):
    # print('test')
    if(len(track.height) >= fps//2):
        if(track.track_id not in fallIds and ((track.height[fps//2]*0.5 > track.height[-1]) 
                                                or (track.height[fps//2]*0.5 > track.height[-1]))):
            fallIds.append(track.track_id)
            return True
    return False 

def publish(source):
    print(source, "PUBLISH!!!!!!!!!!")

    address = map.address()            
    if 'in' in source:
        message = {"count" : incount}
        myMQTTClinet.publish(
            topic = f'/{busNum}/in',
            QoS=1,
            payload= json.dumps(message),
        )
    elif 'out' in source:
        message = {"count" : outcount}
        myMQTTClinet.publish(
            topic = f'/{busNum}/out', 
            QoS=1,
            payload= json.dumps(message),
        )
    else:
        message = {"address" : address, "accidentNum" : fallIdx}
        myMQTTClinet.publish(
            topic = f'/{busNum}/accident', 
            QoS=1,
            payload= json.dumps(message),
        )

def detect(opt, out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, device, model, stride, names, vid_path, dataset, half):
    source = opt.source
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    global HLS_OUTPUT
    global line
    global fallIdx
    global video_width
    global video_height
    global fps
    if "in" in source:  # in 이라는 글자가 포함되면 true
        line = [200, 190, 200, 380]
        HLS_OUTPUT = f'hls/{busNum}/{fallIdx}/' # for test 
    elif "out" in source:
        line = [200, 190, 200, 280]
    else:
        video_width = 1280
        video_height = 720
        fps = 29

    createDirectory(HLS_OUTPUT)

    # # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # # its own .txt file. Hence, in that case, the output folder is not sourrestored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder

    t0 = time.time()
    priorFilesCount = -1
    for (root, directories, files) in os.walk(HLS_OUTPUT):
        priorFilesCount = len(files)

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        if 'fall' in source:
            opt.classes = [0]
        else:
            opt.classes = [1]
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     # add to string
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                tracks = deepsort.tracker.tracks
                # fall detection
                if 'fall' in source:
                    global transmit
                    global transmitFrame
                    print("fallIds: ", fallIds)
                    for track in tracks:
                        if names[int(track.class_id)] == 'person':
                            if transmit != True and isFall(track):
                                HLS_OUTPUT = f'hls/{busNum}/{fallIdx}/'
                                createDirectory(HLS_OUTPUT)
                                ffmpeg_process = run_ffmpeg(720, 480, 6)
                                transmit = True
                                transmitFrame = 0
                                fallIdx += 1
                                break

                # 인원수 카운팅
                else:
                    print('inCountIds:', inCountIds)
                    print('outCountIds:', outCountIds)
                    for track in tracks:
                        # print(track)
                        global incount
                        global outcount
                        if(names[int(track.class_id)] == 'head'): # head를 기준으로 카운트
                            if 'in' in source:  # in count
                                if(track.track_id not in inCountIds and len(track.centroidarr) >= 3
                                and ((track.centroidarr[-3][0] <= line[0]
                                        and track.centroidarr[-3][1] >= line[1]
                                        and track.centroidarr[-1][0] >= line[0]
                                        and abs(track.centroidarr[-1][0] - track.centroidarr[-3][0]) < 360
                                        ) or
                                        (track.centroidarr[-2][0] <= line[0]
                                        and track.centroidarr[-2][1] <= line[1]
                                        and track.centroidarr[-1][0] >= line[0]
                                        and abs(track.centroidarr[-1][0] - track.centroidarr[-2][0]) < 240
                                        ))
                                ):
                                    incount += 1
                                    inCountIds.append(track.track_id)
                            elif 'out' in source:  # out count
                                if(track.track_id not in outCountIds and len(track.centroidarr) >= 3
                                and ((track.centroidarr[-3][0] >= line[0]
                                        and track.centroidarr[-3][1] <= line[3]
                                        and track.centroidarr[-1][0] <= line[0]
                                        and abs(track.centroidarr[-1][0] - track.centroidarr[-3][0]) < 360
                                        ) or
                                        (track.centroidarr[-2][0] >= line[0]
                                        and track.centroidarr[-2][1] <= line[3]
                                        and track.centroidarr[-1][0] <= line[0]
                                        and abs(track.centroidarr[-1][0] - track.centroidarr[-2][0]) < 240
                                        ))
                                ):
                                    outcount += 1
                                    outCountIds.append(track.track_id)


                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        if names[c] == 'head':
                            if id not in inCountIds or id not in outCountIds:
                                annotator.box_label(bboxes, label, color=(153, 255, 153))
                            else:
                                annotator.box_label(bboxes, label, color=(0, 255, 0))

                        if names[c] == 'person':
                            if id not in fallIds:
                                annotator.box_label(bboxes, label, color=(204, 204, 255))
                            else:
                                annotator.box_label(bboxes, label, color=(0, 0, 255))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()

            # 기준 line, text 출력
            text_scale = max(1, im0.shape[1] // 1600)

            if 'in' in source:
                cv2.line(im0, (line[0], line[1]),
                        (line[2], line[3]), (255, 51, 0), 5)
                cv2.putText(im0, 'in: %d' % incount, (20, 20 + text_scale),
                            cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255), thickness=2)
            elif 'out' in source:
                cv2.line(im0, (line[0], line[1]),
                        (line[2], line[3]), (255, 51, 0), 5)
                cv2.putText(im0, 'out: %d' % outcount, (20, 20 + text_scale),
                            cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255), thickness=2)


            # 혼잡도 출력 및 전송#########################################################
            global priorcount
            if 'in' in source and priorcount != incount:
                publish(source)
                priorcount = incount
            
            if 'out' in source and priorcount != outcount:
                publish(source)
                priorcount = outcount

            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            
            if 'fall' in source: 
                if transmit and transmitFrame < fps*savePeriod:
                    if transmitFrame == fps * 13:
                        publish(source)
                    ffmpeg_process.stdin.write(im0)
                    transmitFrame += 1

                if transmitFrame >= fps*savePeriod:
                    print("finish video")
                    ffmpeg_process.stdin.close()
                    transmitFrame = 0
                    transmit = False
                
                for (root, directories, files) in os.walk(HLS_OUTPUT):
                    if len(files) > 0 and priorFilesCount != len(files):
                        priorFilesCount = len(files)
                        for file in files:
                            handle_upload_img(file)
                    else:
                        break


    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str,
                        default='yolov5/weights/crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str,
                        default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='in.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output/' + datetime.now().strftime('%H.%M.%S'),
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")

    with torch.no_grad():
        # detect(args)
        IoTInit()
        S3Init()
        opt = parser.parse_args()
        out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, device, model, stride, names, vid_path, half = trackInit(opt)
        # parameters = ["clip/in1_2_3p.mp4", "clip/out3_1_2p.mp4"]
        parameters = ["clip/out3_1_2p.mp4", "clip/in1_2_3p.mp4", "clip/fall4.mp4"]
        threads = []
        for param in parameters:
            opt.source = param
            source = param
            
            webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
            save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
            txt_file_name = source.split('/')[-1].split('.')[0]
            txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
            opt.img_size = check_img_size(opt.img_size)
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            thread = threading.Thread(target=detect, args=(opt, out, yolo_weights, deep_sort_weights, 
                                                           show_vid, save_vid, save_txt, imgsz, evaluate,
                                                           device, model, stride, names, vid_path, dataset, half
                                                           ))
            thread.start()
            threads.append(thread)
        
        # # 모든 스레드가 종료될 때까지 기다림
        for thread in threads:
            thread.join()


