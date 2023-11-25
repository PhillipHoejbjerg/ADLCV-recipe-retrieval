        for n_images in range(10):
            fig, ax = plt.subplots(2, max_k+1, dpi=200, figsize=(15, 5),tight_layout=True)
            for j in range(2):
                rdn_img_to_plot = n_images+j

                ax[j,0].imshow(denormalize(img[rdn_img_to_plot].permute(1, 2, 0).cpu()))
                ax[j,0].axis('off')
                ax[j,0].set_title(textwrap.fill(self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title, text_width), wrap=True)
                rect = plt.Rectangle((ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[0]), ax[j,0].get_xlim()[1]-ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[1]-ax[j,0].get_ylim()[0],linewidth=5,edgecolor='b',facecolor='none')
                ax[j,0].add_patch(rect)  

                for i in range(max_k):
                    recipe_no = R_top_preds[rdn_img_to_plot][i].item()
                    closest_text = R[recipe_no]
                    closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+recipe_no,:].Title
                    # 274x169
                    wordcloud = WordCloud(background_color='white',width=274,height=169).generate(closest_text)

                    ax[j,i+1].imshow(wordcloud, interpolation='bilinear')
                    ax[j,i+1].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                    ax[j,i+1].axis('off')
                    # make rectangle around this ax
                    if R_positions[rdn_img_to_plot].item() == i:
                        print(f"We should add rectangle at spot {i} in image {n_images}")
                        # Draw rectangle around subbplot
                        rect = plt.Rectangle((ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[0]), ax[j,i+1].get_xlim()[1]-ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[1]-ax[j,i+1].get_ylim()[0],linewidth=5,edgecolor='g',facecolor='none')
                        ax[j,i+1].add_patch(rect)

            plt.savefig(f'reports/figures/im2text_{n_images}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            # plot the closest images
            fig, ax = plt.subplots(2, max_k+1, dpi=200, figsize=(15, 5),tight_layout=True)
            for j in range(2):
                rdn_img_to_plot = n_images+j
                text = R[rdn_img_to_plot]
                wordcloud = WordCloud(background_color='white',width=274,height=169).generate(text)
                closest_text_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+rdn_img_to_plot,:].Title

                ax[j,0].imshow(wordcloud, interpolation='bilinear')
                ax[j,0].axis('off')
                ax[j,0].set_title(textwrap.fill(closest_text_title, text_width), wrap=True)
                rect = plt.Rectangle((ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[0]), ax[j,0].get_xlim()[1]-ax[j,0].get_xlim()[0], ax[j,0].get_ylim()[1]-ax[j,0].get_ylim()[0],linewidth=5,edgecolor='b',facecolor='none')
                ax[j,0].add_patch(rect)                
                
                for i in range(max_k):
                    img_no = img_top_preds[rdn_img_to_plot][i].item()
                    closest_image = img[img_no]
                    closest_image_title = self.test_dataloader_.dataset.csv.iloc[batch_idx*batch_size+img_no,:].Title
                    ax[j,i+1].imshow(denormalize(closest_image.permute(1, 2, 0).cpu()))
                    ax[j,i+1].axis('off')
                    ax[j,i+1].set_title(textwrap.fill(closest_image_title, text_width), wrap=True)
                    # make rectangle around this ax
                    if img_positions[rdn_img_to_plot].item() == i:
                        print(f"We should add rectangle at spot {i} in image {n_images}")
                        # Draw rectangle around subbplot
                        rect = plt.Rectangle((ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[0]), ax[j,i+1].get_xlim()[1]-ax[j,i+1].get_xlim()[0], ax[j,i+1].get_ylim()[1]-ax[j,i+1].get_ylim()[0],linewidth=5,edgecolor='g',facecolor='none')
                        ax[j,i+1].add_patch(rect)

            plt.savefig(f'reports/figures/text2im_{n_images}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close('all')