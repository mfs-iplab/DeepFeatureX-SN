import os, random, csv

working_dir = '/media/lguarnera_group/opontorno/research_activities/DeepFeatureX/'

datasets_path = os.path.join(working_dir, 'datasets')
robustnessdset_path = os.path.join(working_dir, 'testing_robustness')
generalization_path = os.path.join(working_dir, 'testing_generalization')
guidance_path = os.path.join(working_dir, 'guidance.csv')
models_dir = os.path.join(working_dir, 'models/unbalancing-approach')

for folder in ['bm-dm', 'bm-gan', 'bm-real', 'complete']:
    folder_dir = os.path.join(models_dir, folder)
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)

def main():
  with open(guidance_path[:-4]+'.txt', 'w') as f:
      for models_name in os.listdir(datasets_path):
          models_path = os.path.join(datasets_path, models_name)
          for architecture_name in os.listdir(models_path):
              architecture_path = os.path.join(models_path, architecture_name)
              for image in os.listdir(architecture_path):
                  image_path = os.path.join(architecture_path, image)
                  x = random.random()
                  label = int(x >= 0.4) + int(x >= 0.8)
                  f.write(f"{label} & {image_path} & {models_name}\n")

  with open(guidance_path[:-4]+'.txt', 'r') as file:
      data = []
      for line in file:
        label, image, model = line.strip().split(' & ')
        data.append((label, image, model))

  with open(guidance_path, 'w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['label', 'image_path', 'model']) 
      writer.writerows(data)

if __name__=='__main__':
    print('writing new guidance.csv...')
    main()