from ultralytics import YOLO
import torch
import os
torch.cuda.set_device(0)

if __name__ == '__main__': 
    main_folder=os.path.dirname(os.path.realpath(__file__))+'/'
    # Load a model
    #model = YOLO('F:/yolov8/WideCamZF_v1/yolov8m-seg-p2.yaml')  # build a new model from YAML
    model = YOLO(main_folder+"yolo12l-pose.yaml")
    #model = YOLO('yolo12m-pose.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('F:/yolov8/WideCamZF_v1/yolov8n-seg-p2.yaml').load('yolov8n-seg.pt')  # build from YAML and transfer weights
    #model = YOLO('F:/yolov8/WideCamZF_v1/yolov8m-seg-p2.yaml').load('F:/yolov8/WideCamZF_v1/v5train16_p2_best_m.pt')  # build from YAML and transfer weights
    
    #model = YOLO('F:/yolov8/KLGH/data.yaml').load('C:/Users/LBK/runs/segment/train5/weights/best.pt')  # build from YAML and transfer weights



    # Train the model
    model.train(data=main_folder+'data.yaml', epochs=4000, imgsz=640,device=0,patience=40)

    metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="ncnn")  # export the model to ONNX format