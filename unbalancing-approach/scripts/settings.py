import os, random, csv

write=False
datasets_path = '/media/lguarnera_group/opontorno/datasets'
guidance_path = '/home/opontorno/research_activities/overfitting-approach/guidance.csv'
backbones_dir = '/media/lguarnera_group/opontorno/research_activities/overfitting-approach/models'
models_dir = '/media/lguarnera_group/opontorno/research_activities/overfitting-approach/models'
pictures_dir = os.path.expanduser('~/research_activities/overfitting-approach/pictures')

if write:
  with open('/home/opontorno/data/opontorno/datasets_guidance.txt', 'w') as f:
      for models_name in os.listdir(datasets_path):
          models_path = os.path.join(datasets_path, models_name)
          for architecture_name in os.listdir(models_path):
              architecture_path = os.path.join(models_path, architecture_name)
              for image in os.listdir(architecture_path):
                  image_path = os.path.join(architecture_path, image)
                  x = random.random()
                  label = int(x >= 0.4) + int(x >= 0.8)
                  f.write(f"{label} & {image_path} & {models_name}\n")

  with open('/home/opontorno/data/opontorno/datasets_guidance.txt', 'r') as file:
      data = []
      for line in file:
        label, image, model = line.strip().split(' & ')
        data.append((label, image, model))

  with open('/home/opontorno/data/opontorno/datasets_guidance.csv', 'w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['label', 'image_path', 'model']) 
      writer.writerows(data)