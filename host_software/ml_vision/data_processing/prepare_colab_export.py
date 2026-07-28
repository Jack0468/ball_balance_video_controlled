import os
import zipfile
import yaml
import glob
import random


def build_colab_yaml(raw_yaml_path, output_yaml_path):
    with open(raw_yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    colab_yaml = {
        'path': '/content/dataset',
        'train': raw.get('train', []),
        'val': raw.get('val', []),
        'names': raw.get('names', {}),
        'kpt_shape': raw.get('kpt_shape', [4, 3]),
    }

    with open(output_yaml_path, 'w') as f:
        yaml.safe_dump(colab_yaml, f, sort_keys=False)

    print(f'Created {output_yaml_path}')
    return colab_yaml


def add_directory_to_zip(zipf, base_dir, sub_dir):
    abs_dir = os.path.join(base_dir, sub_dir)
    if not os.path.isdir(abs_dir):
        print(f'Warning: directory not found: {abs_dir}')
        return

    for root, dirs, files in os.walk(abs_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, base_dir)
            zipf.write(abs_path, arcname=f'dataset/{rel_path}')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_yaml_path = os.path.join(script_dir, 'raw_dataset.yaml')

    if not os.path.exists(raw_yaml_path):
        raise FileNotFoundError(f'Could not find raw_dataset.yaml at {raw_yaml_path}')

    export_dir = os.path.join(os.path.dirname(script_dir), 'colab_export')
    os.makedirs(export_dir, exist_ok=True)

    colab_yaml_path = os.path.join(export_dir, 'colab_dataset.yaml')
    colab_yaml = build_colab_yaml(raw_yaml_path, colab_yaml_path)

    with open(raw_yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    dataset_root = raw.get('path', '')
    if not dataset_root:
        raise ValueError('The raw_dataset.yaml must contain a valid path field pointing to the dataset root.')
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', dataset_root))

    zip_path = os.path.join(os.path.dirname(script_dir), 'colab_dataset.zip')
    print(f'Creating {zip_path} ... (this may take a minute)')

    # Do not include any background (unlabeled/empty) images in the Colab export
    # Keep only images that have non-empty label files.
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(colab_yaml_path, arcname='dataset/colab_dataset.yaml')

        sources = set(colab_yaml.get('train', []) + colab_yaml.get('val', []))
        for source in sorted(sources):
            images_rel = source  # e.g. '03_yolo_raw_dataset/images'
            images_dir = os.path.join(dataset_root, images_rel)
            if not os.path.isdir(images_dir):
                print(f"Warning: images directory not found: {images_dir}")
                continue

            # collect images
            imgs = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
            if not imgs:
                print(f"Warning: no images found in {images_dir}")
                continue

            # determine labels directory (sibling to images dir)
            parent = os.path.dirname(images_rel)  # e.g. '03_yolo_raw_dataset'
            labels_dir = os.path.join(dataset_root, parent, 'labels')

            labeled_imgs = []
            background_imgs = []
            for img in imgs:
                base = os.path.splitext(os.path.basename(img))[0]
                lbl = os.path.join(labels_dir, base + '.txt')
                if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                    labeled_imgs.append(img)
                else:
                    background_imgs.append(img)

            # Add only labeled images (exclude all background/unlabeled images)
            to_add = sorted(labeled_imgs)
            print(f"Adding {len(to_add)} labeled files from {images_rel} (labels: {len(labeled_imgs)}, backgrounds excluded: {len(background_imgs)})")
            for abs_path in to_add:
                rel_path = os.path.relpath(abs_path, dataset_root)
                zipf.write(abs_path, arcname=f'dataset/{rel_path}')

                # also add corresponding label file (should exist for these images)
                base = os.path.splitext(os.path.basename(abs_path))[0]
                label_abs = os.path.join(labels_dir, base + '.txt')
                if os.path.exists(label_abs):
                    label_rel = os.path.relpath(label_abs, dataset_root)
                    zipf.write(label_abs, arcname=f'dataset/{label_rel}')

    print(f'Done! You can now upload {zip_path} to Google Drive.')


if __name__ == '__main__':
    main()
