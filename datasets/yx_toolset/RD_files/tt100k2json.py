import json
import os
import cv2

# Take care! For automatic data generation, we need to write another script contains some terminal commands 99... 

json_add = "/media/yingges/Data/Datasets/TT100K_TS/data/annotations.json"    # the address of annotations.json of tt100k
label_add = "/media/yingges/Data/Datasets/TT100K_TS/data/"            # the label address of the train/test/other set
mode_list = ["train/", "test/", "other/"]
json_save_path = "/home/yingges/Downloads/tt100kjson/"
new_jpg_save_path = "/home/yingges/Downloads/tt100kjpg/"

# write_object_txt function works to extract necesaary infos from 
def write_object_txt(json_add = json_add, label_add = label_add, mode_list = mode_list):
	tt100k_json_file = json.loads(open(json_add).read())  # load all the contexts of tt100k annos.
	types_list = tt100k_json_file["types"]  # all traffic-sign types in tt100k dataset
	index = 0
	for mode in mode_list:

		ids_txt = open(label_add + mode.split("/")[0] + "_ids.txt", 'r').readlines()
		print "the length of the {}_ids.txt is {}".format(mode, len(ids_txt))
		
		for item in ids_txt:
			out_json_dict = {"path": "", "outputs": {"object": []}, "time_labeled": 1574423089041, "labeled": "true",
							 "size": {"width": 2048, "height": 2048, "depth": 3}}
			temp = tt100k_json_file["imgs"][
				str(item).rstrip('\n')]  # Take care. There the image_id should be str(numbers).
			print(index)
			temp_path = temp["path"]
			out_json_dict['path'] = str(index)+".jpg"			#print(temp_path)
			temp_objects = temp["objects"]  # every image may contain several traffic-sign objects...
			#print(temp_objects)
			object_id = 0
			for object_ in temp_objects:
				object_dict = {"name": "", "bndbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}}
				bbox = object_["bbox"]
				#print(bbox)
				object_dict["name"] = object_["category"]
				object_dict["bndbox"]["xmin"], object_dict["bndbox"]["ymin"], object_dict["bndbox"]["xmax"], object_dict["bndbox"]["ymax"] = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
				out_json_dict["outputs"]["object"].append(object_dict)
				# If the trafficsign class is not in the basic types of tt100k, we should set it as the none-existed class.
				# But remember in the following sections, we will regenerate the classnumber for classification...
				index += 1
				object_id += 1  # If we print this variable, we can get how many objects in the current image.
			if(object_id !=0):
				img = cv2.imread(os.path.join(label_add,temp_path))
				if img is not  None:
					cv2.imwrite(new_jpg_save_path + str(index)+".jpg", img)
					out_file = open(os.path.join(json_save_path,str(index)+".json"), 'w')
					out_file.write(json.dumps(out_json_dict))
					out_file.close()

	print(index)




write_object_txt()


