from src.data.dataloader import *
import argparse
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPTextConfig, CLIPVisionConfig



if __name__ == "__main__":
    # Get pretained CLIP models.
    #model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # If we want to change the config:
    config_text = CLIPTextConfig(max_position_embeddings=2048) #Helps max_tokens I think?
    config_vision = CLIPVisionConfig() # This is no issue
    config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
    model = CLIPModel(config)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    

    parser = argparse.ArgumentParser(description='Recipe Retrieval CLIP Evaluation Script')
    parser.add_argument('--center_crop', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--text_encoder_name', type=str, default="roberta_base", help='roberta_base, transformer_base')
    parser.add_argument("--text_mode", type=str, default='title ingredients instructions', help="text mode - default title")
    parser.add_argument('--num_workers', type=int, default=11, help='number of workers - default 0')
    parser.add_argument('--loss_fn', type=str, default="ClipLoss", help='Loss_fn - default cosine')
    parser.add_argument('--batch_size',    type=int, default=8, help='batch size - default 64')
    args = parser.parse_args()

    #data_set = CombinedDataSet(p=0.2, mode='test', text=['title'], yield_raw_text=True, args=args)
    #data_loader = DataLoader(data_set, batch_size=8, shuffle=True, collate_fn=collate_batch_text_roberta)

    data_loader = get_dataloader(args, mode = 'train')

    # This is just for debugging, it can be done way better probably. Or we can just do it in the loop...

    catcher = [] # [ [img_0 (PIL_IMG), text_0 (str)],  [img_1 (PIL_IMG), text_1 (str)], ..., [img_N (PIL_IMG), text_N (str)] ]

    for img, text, _ in data_loader:
        for i in range(8):
            catcher.append([to_pil_image(img[i]), text[i]])
            break
        break
    inputs = processor(text=catcher[0][1], images=catcher[0][0], return_tensors="pt", padding=True)

    outputs = model(**inputs)
    print(f'Text emb_dim from CLIP:\n{outputs.text_embeds.shape}') # [B x 512]
    print(f'Image emb_dim from CLIP:\n{outputs.image_embeds.shape}') # [B x 512]

    # from guide
    #logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities