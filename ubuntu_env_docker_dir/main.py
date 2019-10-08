# # This is a temporary fix.
# # Find a way to add this permanantly to the sys.path.
# import sys
# sys.path.append("./packages")

# import cntk
# import caffe2.python.onnx.backend
# from caffe2.python import core, workspace

# import numpy as np

# make input Numpy array of correct dimensions and type as required by the model

# model_file = onnx.load('./mnist/model.onnx')
# #
# inputArray = np.zeros((1,1,28,28))
# #print(inputArray)
# opset = model_file.opset_import.add()
# opset.domain = ''
# opset.version = 2
# output = caffe2.python.onnx.backend.run_model(model_file, inputArray.astype(np.float32))

# from onnx import AttributeProto

# model = onnx.load('./mnist/model.onnx')
# node = model.graph.node[0] #use the first node of op_type as "Conv"
# # node.attribute[0].type = AttributeProto.INT
# onnx.checker.check_node(node)

# import onnxruntime as rt
# import numpy
# from onnxruntime.datasets import get_example

# example1 = get_example('./mnist/model.onnx')
# sess = rt.InferenceSession(example1)


# import onnxruntime as rt
# import numpy
# sess = rt.InferenceSession("./mnist/model.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# pred = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]


# Inference in Caffe2 using the ONNX model
# import caffe2.python.onnx.backend as backend
# import onnx
# import 
 
# # First load the onnx model
# model = onnx.load("animals_caltech.onnx")
 
# # Prepare the backend
# rep = backend.prepare(model, device="CPU")
 
# # Transform the image
# transform = transforms.Compose([
#         transforms.Resize(size=224),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
 
# # Load and show the image
# test_image_name = "giraffe.jpg"
# test_image = Image.open(test_image_name)
# display(test_image)
 
# # Apply the transformations to the input image and convert it into a tensor
# test_image_tensor = transform(test_image)
 
# # Make the input image ready to be input as a batch of size 1
# test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
 
# # Convert the tensor to numpy array
# np_image = test_image_tensor.numpy()
 
# # Pass the numpy array to run through the ONNX model
# outputs = rep.run(np_image.astype(np.float32))
 
# # Dictionary with class name and index
# idx_to_class = {0: 'bear', 1: 'chimp', 2: 'giraffe', 3: 'gorilla', 4: 'llama', 5: 'ostrich', 6: 'porcupine', 7: 'skunk', 8: 'triceratops', 9: 'zebra'}
 
# ps = torch.exp(torch.from_numpy(outputs[0]))
# topk, topclass = ps.topk(10, dim=1)
# for i in range(10):
#     print("Prediction", '{:2d}'.format(i+1), ":", '{:11}'.format(idx_to_class[topclass.cpu().numpy()[0][i]]), ", Class Id : ", topclass[0][i].numpy(), " Score: ", topk.cpu().detach().numpy()[0][i])




# import onnxruntime as rt
# import numpy
# sess = rt.InferenceSession("./mnist/model.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
# print(pred_onx)


# apt-get upgrade python3
# apt-get install python3 -- Y
# apt-get install build-essential libssl-dev libffi-dev python-dev --Y
# apt install python3-pip --Y

# import cntk as C
# import numpy as np
# from PIL import Image
# from IPython.core.display import display
# import pickle

# # Import the model into CNTK via CNTK's import API
# z = C.Function.load("vgg16.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)
# print("Loaded vgg16.onnx!")
# img = Image.open("Siberian_Husky_bi-eyed_Flickr.jpg")
# img = img.resize((224,224))
# rgb_img = np.asarray(img, dtype=np.float32) - 128
# bgr_img = rgb_img[..., [2,1,0]]
# img_data = np.ascontiguousarray(np.rollaxis(bgr_img,2))
# predictions = np.squeeze(z.eval({z.arguments[0]:[img_data]}))
# top_class = np.argmax(predictions)
# print(top_class)
# labels_dict = pickle.load(open("imagenet1000_clsid_to_human.pkl", "rb"))
# print(labels_dict[top_class])


