# uncomment a block to run a experiment

######### CVC_Colon - CVC_Clinic ########
train_dataset = "CVC_Colon"
test_dataset = "CVC_Clinic"
multiply_value = 255
threshold_segmentation = 0.5
encoder="resnest101"
couple_unet=True
decoder_attention_type=None
path_load_checkpoint = None
architure = "Attention-CUNet"
loss_function = "tverskyloss"
weights = "imagenet"
image_dir = "images"
mask_dir = "masks"
test_path = "./datasets/{0}".format(test_dataset)
path_checkpoint = "./checkpoints/Attention-CUNet_resnest101_CVC_Colon_CVC_Clinic_tverskyloss.pth"


######### CVC_Clinic - ETIS ########
#train_dataset = "CVC_Clinic"
#test_dataset = "ETIS"
#multiply_value = 255
#threshold_segmentation = 0.5
#encoder="resnest101"
#couple_unet = True
#decoder_attention_type = None
#path_load_checkpoint = None
#architure = "Attention-CUNet"
#loss_function = "tverskyloss"
#weights = "imagenet"
#image_dir = "images"
#mask_dir = "masks"
#test_path = "./datasets/{0}".format(test_dataset)
#path_checkpoint = "./checkpoints/Attention-CUNet_resnest101_CVC_Clinic_ETIS_tverskyloss.pth"


######## Pranet########
#train_dataset = "pranet"
#test_dataset = "pranet"
#multiply_value = 1
#threshold_segmentation = 0.5
#encoder="resnest101"
#couple_unet = True
#decoder_attention_type = None
#path_load_checkpoint = None
#architure = "Attention-CUNet"
#loss_function = "tverskyloss"
#weights = "imagenet"
#image_dir = "images"
#mask_dir = "masks"
#test_path = "./datasets/Pranet/Kvasir"
#path_checkpoint = "./checkpoints/Attention-CUNet_resnest101_pranet_pranet_tverskyloss.pth"

