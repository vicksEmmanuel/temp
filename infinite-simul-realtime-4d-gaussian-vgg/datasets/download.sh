# Neural 3D Dataset
# Download the dataset from the official repository
# https://github.com/facebookresearch/Neural_3D_Video

wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cut_roasted_beef.zip
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z01
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z02
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z03
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.zip
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
# wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/sear_steak.zip


mkdir -p ./data
mkdir -p ./data/Neural_3D_Dataset

unzip coffee_martini.zip -d ./data/Neural_3D_Dataset
rm coffee_martini.zip

# unzip cook_spinach.zip -d ./data/Neural_3D_Dataset
# rm cook_spinach.zip

# unzip cut_roasted_beef.zip -d ./data/Neural_3D_Dataset
# rm cut_roasted_beef.zip

# cat flame_salmon_1_split.z01 flame_salmon_1_split.z02 flame_salmon_1_split.z03 > flame_salmon_1_split.zip
# unzip flame_salmon_1_split.zip -d ./data/Neural_3D_Dataset
# rm flame_salmon_1_split.zip
# rm flame_salmon_1_split.z01
# rm flame_salmon_1_split.z02
# rm flame_salmon_1_split.z03

# unzip flame_steak.zip -d ./data/Neural_3D_Dataset
# rm flame_steak.zip

# unzip sear_steak.zip -d ./data/Neural_3D_Dataset
# rm sear_steak.zip




# Immersive Light Field Video with a Layered Mesh Representation
# Download the dataset from the official repository
# https://github.com/augmentedperception/deepview_video_dataset?tab=readme-ov-file

# wget https://storage.googleapis.com/deepview_video_raw_data/15_Branches.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/14_Puppy.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/13_Birds.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/12_Cave.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/11_Alexa_Meade_Face_Paint_2.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/10_Alexa_Meade_Face_Paint_1.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/09_Alexa_Meade_Exhibit.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/08_Pond.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/07_Car.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/06_Goats.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/05_Horse.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/04_Truck.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/03_Dog.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/02_Flames.zip
# wget https://storage.googleapis.com/deepview_video_raw_data/01_Welder.zip

# mkdir -p ./data/Immersive_Light_Field_Video

# unzip 15_Branches.zip -d ./data/Immersive_Light_Field_Video
# rm 15_Branches.zip

# unzip 14_Puppy.zip -d ./data/Immersive_Light_Field_Video
# rm 14_Puppy.zip

# unzip 13_Birds.zip -d ./data/Immersive_Light_Field_Video
# rm 13_Birds.zip

# unzip 12_Cave.zip -d ./data/Immersive_Light_Field_Video
# rm 12_Cave.zip

# unzip 11_Alexa_Meade_Face_Paint_2.zip -d ./data/Immersive_Light_Field_Video
# rm 11_Alexa_Meade_Face_Paint_2.zip

# unzip 10_Alexa_Meade_Face_Paint_1.zip -d ./data/Immersive_Light_Field_Video
# rm 10_Alexa_Meade_Face_Paint_1.zip

# unzip 09_Alexa_Meade_Exhibit.zip -d ./data/Immersive_Light_Field_Video
# rm 09_Alexa_Meade_Exhibit.zip

# unzip 08_Pond.zip -d ./data/Immersive_Light_Field_Video
# rm 08_Pond.zip

# unzip 07_Car.zip -d ./data/Immersive_Light_Field_Video
# rm 07_Car.zip

# unzip 06_Goats.zip -d ./data/Immersive_Light_Field_Video
# rm 06_Goats.zip

# unzip 05_Horse.zip -d ./data/Immersive_Light_Field_Video
# rm 05_Horse.zip

# unzip 04_Truck.zip -d ./data/Immersive_Light_Field_Video
# rm 04_Truck.zip

# unzip 03_Dog.zip -d ./data/Immersive_Light_Field_Video
# rm 03_Dog.zip

# unzip 02_Flames.zip -d ./data/Immersive_Light_Field_Video
# rm 02_Flames.zip

# unzip 01_Welder.zip -d ./data/Immersive_Light_Field_Video
# rm 01_Welder.zip


python3 datasets/preprocessor.py