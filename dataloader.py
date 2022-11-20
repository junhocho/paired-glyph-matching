import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class ImageAttr(data.Dataset):
    """Dataset class for the ImageAttr dataset."""
    def __init__(self, image_dir, attr_path, transform, mode,
                 binary=False, n_style=4,
                 char_num=52, unsuper_num=968, train_num=120, val_num=28):
        """Initialize and preprocess the ImageAttr dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.n_style = n_style

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.super_train_dataset = []
        self.super_test_dataset = []
        self.unsuper_train_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.char_num = char_num
        self.unsupervised_font_num = unsuper_num
        self.train_font_num = train_num
        self.val_font_num = val_num

        self.test_super_unsuper = {}
        for super_font in range(self.train_font_num+self.val_font_num):
            self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)

        self.char_idx_offset = 10

        self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]

        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.super_train_dataset) + len(self.unsuper_train_dataset)
        else:
            self.num_images = len(self.super_test_dataset)

    def preprocess(self):
        """Preprocess the font attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]

        train_size = self.char_num * self.train_font_num
        val_size = self.char_num * self.val_font_num

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            target_char = filename.split('/')[1].split('.')[0]
            char_class = int(target_char) - self.char_idx_offset
            font_class = int(i / self.char_num)

            attr_value = []
            for val in values:
                if self.binary:
                    attr_value.append(val == '1')
                else:
                    attr_value.append(eval(val) / 100.0)

            # print(filename, char_class, font_class, attr_value)

            if i < train_size:
                self.super_train_dataset.append([filename, char_class, font_class, attr_value])
            elif i < train_size + val_size:
                self.super_test_dataset.append([filename, char_class, font_class, attr_value])
            else:
                self.unsuper_train_dataset.append([filename, char_class, font_class, attr_value])

        print('Finished preprocessing the Image Attribute (Explo) dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # dataset = self.super_train_dataset if self.mode == 'train' else self.super_test_dataset

        if self.mode == 'train':
            if index < len(self.super_train_dataset):
                filename_A, charclass_A, fontclass_A, attr_A = self.super_train_dataset[index]
                label_A = 1.0
                font_embed_A = self.unsupervised_font_num  # dummy id 968
                # B is supervised or unsupervised
                sample_p = random.random()
                if sample_p < 0.5:
                    # Unsupervise
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.unsupervised_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.unsuper_train_dataset[index_B]
                    label_B = 0.0
                    font_embed_B = fontclass_B - self.train_font_num - self.val_font_num  # convert to [0, 967]
                else:
                    # Supervise
                    # get B from supervise train !!
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.train_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.super_train_dataset[index_B]
                    label_B = 1.0
                    font_embed_B = self.unsupervised_font_num  # dummy id 968

            else:
                # get A from unsupervise train !!
                index = index - len(self.super_train_dataset)
                filename_A, charclass_A, fontclass_A, attr_A = self.unsuper_train_dataset[index]
                label_A = 0.0
                font_embed_A = fontclass_A - self.train_font_num - self.val_font_num
                # B is supervised or unsupervised
                sample_p = random.random()
                if sample_p < 0.5:
                    # Unsupervise
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.unsupervised_font_num - 1)  # noqa
                    filename_B, charclass_B, fontclass_B, attr_B = self.unsuper_train_dataset[index_B]
                    label_B = 0.0
                    font_embed_B = fontclass_B - self.train_font_num - self.val_font_num  # convert to [0, 967]
                else:
                    # Supervise
                    # get B from supervise train !!
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.train_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.super_train_dataset[index_B]
                    label_B = 1.0
                    font_embed_B = self.unsupervised_font_num  # dummy id 968

        else:
            # load the random one from unsupervise data as the reference aka A
            # unsuper to super
            font_index_super = index // self.char_num + self.train_font_num
            font_index_unsuper = self.test_super_unsuper[font_index_super]
            char_index_unsuper = index % self.char_num + self.char_num * font_index_unsuper
            filename_A, charclass_A, fontclass_A, attr_A = self.unsuper_train_dataset[char_index_unsuper]
            label_A = 0.0
            font_embed_A = fontclass_A - self.train_font_num - self.val_font_num  # convert to [0, 967]

            filename_B, charclass_B, fontclass_B, attr_B = self.super_test_dataset[index]
            label_B = 1.0
            font_embed_B = self.unsupervised_font_num  # dummy id 968

        # Get style samples
        random.shuffle(self.chars)
        style_chars = self.chars[:self.n_style]
        styles_A = []
        if self.n_style == 1:
            styles_A.append(filename_A)
        else:
            for char in style_chars:
                styles_A.append(rreplace(filename_A, str(charclass_A+10), str(char), 1))

        # random.shuffle(self.chars)
        # style_chars = self.chars[:self.n_style]
        chars_without_A = [c for c in self.chars if c!=charclass_A]
        random.shuffle(chars_without_A)
        style_chars = chars_without_A[:self.n_style]
        assert(not charclass_A in style_chars)

        styles_B = []
        if self.n_style == 1:
            styles_B.append(filename_B)
        else:
            for char in style_chars:
                styles_B.append(rreplace(filename_B, str(charclass_B+10), str(char), 1))

        image_A = Image.open(os.path.join(self.image_dir, filename_A)).convert('RGB')
        image_B = Image.open(os.path.join(self.image_dir, filename_B)).convert('RGB')
        # Open and transform style images
        style_imgs_A = []
        for style_A in styles_A:
            style_imgs_A.append(self.transform(Image.open(os.path.join(self.image_dir, style_A)).convert('RGB')))
        style_imgs_A = torch.cat(style_imgs_A)
        style_imgs_B = []
        for style_B in styles_B:
            style_imgs_B.append(self.transform(Image.open(os.path.join(self.image_dir, style_B)).convert('RGB')))
        style_imgs_B = torch.cat(style_imgs_B)

        return {"img_A": self.transform(image_A), "charclass_A": torch.LongTensor([charclass_A]),
                "fontclass_A": torch.LongTensor([fontclass_A]), "attr_A": torch.FloatTensor(attr_A),
                "styles_A": style_imgs_A,
                "fontembed_A": torch.LongTensor([font_embed_A]),
                "label_A": torch.FloatTensor([label_A]),
                "img_B": self.transform(image_B), "charclass_B": torch.LongTensor([charclass_B]),
                "fontclass_B": torch.LongTensor([fontclass_B]), "attr_B": torch.FloatTensor(attr_B),
                "styles_B": style_imgs_B,
                "fontembed_B": torch.LongTensor([font_embed_B]),
                "label_B": torch.FloatTensor([label_B])}

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, image_size=256,
               batch_size=16, dataset_name='explor_all', mode='train', num_workers=8,
               binary=False, n_style=4,
               char_num=52, unsuper_num=968, train_num=120, val_num=28):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset_name == 'explor_all':
        dataset = ImageAttr(image_dir, attr_path, transform,
                            mode, binary, n_style,
                            char_num=52, unsuper_num=968,
                            train_num=120, val_num=28)
    data_loader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)

    return data_loader


"""
JH implementing
"""

class OneGlyphPerFont_Donovan(data.Dataset):
    """Dataset class for the ImageAttr dataset."""
    def __init__(self, image_dir, attr_path, transform, mode,
                 binary=False,
                 unsuper_num=968, train_num=120, val_num=28, char_set='alphabets'):
        """Initialize and preprocess the ImageAttr dataset."""

        self.dataset_name = "Donovan"
        
        self.image_dir = image_dir
        self.attr_path = attr_path
        if not 'alphanumeric' in attr_path :
            raise RuntimeError("use different attr file : {}".format(attr_path))
        self.use_attr = True

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.super_train_dataset = []
        self.super_test_dataset = []
        self.unsuper_train_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        """
        01234567789abcdefg...xyzABCDE...XYZ
        """
        numbers = "0123456789"
        lowercases = "abcdefghijklmnopqrstuvwxyz"
        uppercases = lowercases.upper()

        if char_set == 'alphabets':
            self.char_num = 52
            self.char_idx_offset = 10
            self.idx2chars = lowercases+uppercases
            print("Load alphabets")
        elif char_set == 'capitals':
            self.char_num = 26
            self.char_idx_offset = 10 + 26
            self.idx2chars = uppercases
            print("Load capitals")
        elif char_set == 'numbers':
            self.char_num = 10
            self.char_idx_offset = 0
            self.idx2chars = numbers
            print("Load numbers")
        elif char_set == 'alphanumeric':
            self.char_num = 62
            self.char_idx_offset = 0
            self.idx2chars = numbers+lowercases+uppercases
            print("Load alphanumeric")


        self.unsupervised_font_num = unsuper_num
        self.train_font_num = train_num
        self.val_font_num = val_num


        self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]

        self.fontcls2fontname = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = self.unsupervised_font_num + self.train_font_num
        elif mode == 'test':
            self.num_images = self.val_font_num
        elif mode == 'train_test':
            self.num_images = self.train_font_num + self.val_font_num
        else:
            raise NotImplementedError("what mode is this? :", mode)



    def preprocess(self):
        """Preprocess the font attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]

        train_size = 62 * self.train_font_num
        val_size = 62 * self.val_font_num



        # We will group glyphs by fonts.
        curr_font_data = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            target_char = filename.split('/')[1].split('.')[0]
            if int(target_char) < self.char_idx_offset:
                continue
            elif int(target_char) > self.char_idx_offset + self.char_num - 1:
                continue
            char_class = int(target_char) - self.char_idx_offset
            font_class = int(i/62) # int(i / self.char_num) 

            attr_value = []
            for val in values:
                if self.binary:
                    attr_value.append(val == '1')
                else:
                    attr_value.append(eval(val) / 100.0)


            ## initally, data listed as : super_train, super_test, unsuper_train
            if font_class  > self.val_font_num + self.train_font_num -1:
                ## unsuper_train
                font_class -= self.val_font_num
            elif font_class > self.train_font_num - 1:
                ## super_test
                font_class += self.unsupervised_font_num
            self.fontcls2fontname[font_class] = filename.split('/')[0]

            curr_glyph_data = [filename, char_class, font_class, attr_value]
            curr_font_data.append(curr_glyph_data)
            if (len(curr_font_data) == len(self.chars)):  # collected all glyphs for a font
                if i < train_size:
                    self.super_train_dataset.append(curr_font_data)
                elif i < train_size + val_size:
                    self.super_test_dataset.append(curr_font_data)
                else:
                    self.unsuper_train_dataset.append(curr_font_data)
                curr_font_data = []

        print("# of Supervised train dataset:", len(self.super_train_dataset))
        print("# of Unsupervised train dataset:", len(self.unsuper_train_dataset))
        print("# of Supervised test dataset:", len(self.super_test_dataset))
        print('Finished preprocessing the Donovan data (per char) for embedding task')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        # print(index, self.train_font_num)
        if self.mode == 'train':
            if index < self.train_font_num:
                font_A = self.super_train_dataset[index]
                label_A = 1.0
            else:
                # get A from unsupervise train !!
                index = index - self.train_font_num
                font_A = self.unsuper_train_dataset[index]
                label_A = 0.0

            num_glyphs = len(font_A)
            indices = list(range(len(font_A)))
            random.shuffle(indices)  # each __get_item samples two different glyphs
            i = indices[0]  # random sample two glyphs

            glyph_i = font_A[i]
        elif self.mode in ['test', 'train_test']:
            font_idx = index // self.char_num
            char_idx = index % self.char_num

            if self.mode == 'test':
                font_A = self.super_test_dataset[font_idx]
            elif self.mode == 'train_test':
                if font_idx < self.train_font_num:
                    font_A = self.super_train_dataset[font_idx]
                else:
                    font_idx -= self.train_font_num
                    font_A = self.super_test_dataset[font_idx]
            glyph_i = font_A[char_idx]
            label_A = 1.0

        filename_i, charclass_i, fontclass_i, attr_i = glyph_i

        image_i = Image.open(os.path.join(self.image_dir, filename_i)).convert('RGB')

        return {"img_i": self.transform(image_i), "charclass_i": torch.LongTensor([charclass_i]),
                "fontclass": torch.LongTensor([fontclass_i]), "attr": torch.FloatTensor(attr_i),
                "label_A": torch.FloatTensor([label_A]),
                }

    def __len__(self):
        """Return the number of images."""
        if self.mode == 'train':
            return self.num_images
        elif self.mode == 'test':
            return self.char_num * len(self.super_test_dataset)
        elif self.mode == 'train_test':
            return self.char_num * (len(self.super_train_dataset) + len(self.super_test_dataset))



class TwoGlyphsPerFont_Donovan(data.Dataset):
    """Dataset class for the ImageAttr dataset."""
    def __init__(self, image_dir, attr_path, transform, mode,
                 binary=False,
                 char_num=52, unsuper_num=968, train_num=120, val_num=28):
        """Initialize and preprocess the ImageAttr dataset."""
        
        self.image_dir = image_dir
        self.attr_path = attr_path

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.super_train_dataset = []
        self.super_test_dataset = []
        self.unsuper_train_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.idx2chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        """
        01234567789abcdefg...xyzABCDE...XYZ
        """

        self.char_num = char_num
        self.unsupervised_font_num = unsuper_num
        self.train_font_num = train_num
        self.val_font_num = val_num

        # self.test_super_unsuper = {}
        # for super_font in range(self.train_font_num+self.val_font_num):
        #     self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)

        self.char_idx_offset = 10

        self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]

        self.preprocess()

        assert(self.mode == 'train')
        if mode == 'train':
            self.num_images = self.unsupervised_font_num + self.train_font_num
        elif mode == 'test':
            self.num_images = self.val_font_num
            self.i = 42
            self.j = 43
        else:
            raise NotImplementedError("what mode is this? :", mode)

    def preprocess(self):
        """Preprocess the font attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]

        train_size = self.char_num * self.train_font_num
        val_size = self.char_num * self.val_font_num



        # We will group glyphs by fonts.
        curr_font_data = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            target_char = filename.split('/')[1].split('.')[0]
            char_class = int(target_char) - self.char_idx_offset
            font_class = int(i / self.char_num)

            attr_value = []
            for val in values:
                if self.binary:
                    attr_value.append(val == '1')
                else:
                    attr_value.append(eval(val) / 100.0)

            ## initally, data listed as : super_train, super_test, unsuper_train
            if font_class  > self.val_font_num + self.train_font_num -1:
                font_class -= self.val_font_num
            elif font_class > self.train_font_num - 1:
                font_class += self.unsupervised_font_num

            # print(filename, char_class, font_class, attr_value)
            # print(filename, char_class, font_class)
            curr_glyph_data = [filename, char_class, font_class, attr_value]
            curr_font_data.append(curr_glyph_data)
            if (len(curr_font_data) == len(self.chars)):  # collected all glyphs for a font
                if i < train_size:
                    self.super_train_dataset.append(curr_font_data)
                elif i < train_size + val_size:
                    self.super_test_dataset.append(curr_font_data)
                else:
                    self.unsuper_train_dataset.append(curr_font_data)
                curr_font_data = []

        print("# of Supervised train dataset:", len(self.super_train_dataset))
        print("# of Unsupervised train dataset:", len(self.unsuper_train_dataset))
        print("# of Supervised test dataset:", len(self.super_test_dataset))
        print('Finished preprocessing the Donovan data for embedding task')

    def change_infer_char(self, i, j):
        self.i = i
        self.j = j

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        # print(index, self.train_font_num)
        if self.mode == 'train':
            if index < self.train_font_num:
                font_A = self.super_train_dataset[index]
                label_A = 1.0
            else:
                # get A from unsupervise train !!
                index = index - self.train_font_num
                font_A = self.unsuper_train_dataset[index]
                label_A = 0.0

            num_glyphs = len(font_A)
            indices = list(range(len(font_A)))
            random.shuffle(indices)  # each __get_item samples two different glyphs
            i = indices[0]  # random sample two glyphs
            j = indices[1]  # random sample two glyphs
        else:
            # for validation....
            # we use only two representative glyphs? for say, P , Q 
            font_A = self.super_test_dataset[index]
            i = self.i # Q
            j = self.j # R
            label_A = 1.0


        glyph_i = font_A[i]
        glyph_j = font_A[j]

        filename_i, charclass_i, fontclass_i, attr_i = glyph_i
        filename_j, charclass_j, fontclass_j, attr_j = glyph_j

        assert(fontclass_i == fontclass_j), "fontclass should match"
        assert(charclass_i != charclass_j), "charclass should be different"

        image_i = Image.open(os.path.join(self.image_dir, filename_i)).convert('RGB')
        image_j = Image.open(os.path.join(self.image_dir, filename_j)).convert('RGB')

        return {"img_i": self.transform(image_i), "charclass_i": torch.LongTensor([charclass_i]),
                "img_j": self.transform(image_j), "charclass_j": torch.LongTensor([charclass_j]),
                "fontclass": torch.LongTensor([fontclass_i]), "attr": torch.FloatTensor(attr_i),
                "label_A": torch.FloatTensor([label_A]),
                }

    def __len__(self):
        """Return the number of images."""
        return self.num_images

"""
OFL DATASET
"""
class OneGlyphPerFont(data.Dataset): 
    """Dataset class for the OFL dataset."""
    def __init__(self, dataset_name, image_dir, attr_path, transform, mode,
                 binary=False,
                 char_set='alphabets'):
        """Initialize and preprocess the ImageAttr dataset."""

        assert(dataset_name in ['OFL', 'Capitals64'])
        self.dataset_name = dataset_name
        
        self.image_dir = image_dir
        self.attr_path = attr_path # XXX no attr, this used as glyph path. consider name change.
        self.use_attr = False

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []


        """
        01234567789abcdefg...xyzABCDE...XYZ
        """
        # self.test_super_unsuper = {}
        # for super_font in range(self.train_font_num+self.val_font_num):
        #     self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)

        numbers = "0123456789"
        lowercases = "abcdefghijklmnopqrstuvwxyz"
        uppercases = lowercases.upper()
        if char_set == 'alphabets':
            self.char_num = 52
            self.char_idx_offset = 10
            self.idx2chars = lowercases+uppercases
            print("Load alphabets")
        elif char_set == 'capitals':
            self.char_num = 26
            self.char_idx_offset = 10 + 26
            self.idx2chars = uppercases
            print("Load capitals")
        elif char_set == 'numbers':
            self.char_num = 10
            self.char_idx_offset = 0
            self.idx2chars = numbers
            print("Load number")
        elif char_set == 'alphanumeric':
            self.char_num = 62
            self.char_idx_offset = 0
            self.idx2chars = numbers+lowercases+uppercases
            print("Load alphanumeric")


        if dataset_name == 'OFL':
            self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]
            self.preprocess_OFL()

        elif dataset_name == 'Capitals64':
            self.idx2chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            assert(self.char_num == 26)
            self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]
            self.preprocess_Capitals64()


    def preprocess_Capitals64(self):
        if self.mode == 'train':
            self.image_dir = os.path.join(self.image_dir, 'train_split')
        elif self.mode == 'val':
            self.image_dir = os.path.join(self.image_dir, 'val_split')
        elif self.mode == 'test':
            self.image_dir = os.path.join(self.image_dir, 'test_split')
        else:
            raise NotImplementedError('Unknown portion'.format(self.mode))

        font_names = os.listdir(self.image_dir)
        attr_value = [0. for i in range(37) ] ## NOTE :dummy
        dataset = []
        for ii, font_class in enumerate(font_names):
            # font_path = os.path.join(self.image_dir, font_class)
            curr_font_data = []
            for char_class in range(self.char_num):
                char = self.idx2chars[char_class]
                filename = font_class + '/' + char + '.png'
                curr_glyph_data = [filename, char_class, ii, attr_value, font_class]
                curr_font_data.append(curr_glyph_data)
            dataset.append(curr_font_data)

        print('Dataset mode : {}'.format(self.mode))
        print('Finished preprocessing the Capitals64 dataset... {} fonts'.format(len(dataset)))
        if self.mode == 'train':
            self.train_dataset = dataset
            self.train_font_num = len(self.train_dataset)
            assert(self.train_font_num == 7649)
            print('Trainset : {} fonts'.format(len(self.train_dataset)))
        elif self.mode == 'val':
            self.val_dataset = dataset
            self.val_font_num = len(self.val_dataset)
            assert(self.val_font_num == 1473)
            print('Valset : {} fonts'.format(len(self.val_dataset)))
        elif self.mode == 'test':
            self.test_dataset = dataset
            self.test_font_num = len(self.test_dataset)
            assert(self.test_font_num == 1560)
            print('Testset : {} fonts'.format(len(self.test_dataset)))



    def preprocess_OFL(self):
        train_num=3702
        val_num=100
        test_num=100
        self.train_font_num = train_num
        self.val_font_num = val_num
        self.test_font_num = test_num
        # XXX HARDCODED atm
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        curr_font_data = []
        dataset = []
        for i, line in enumerate(lines):
            filename = line
            fontname = filename.split('/')[0]
            char_class = i % 62
            if char_class < self.char_idx_offset: #XXX  hardcoded excluding nums
                continue
            elif char_class > self.char_idx_offset + self.char_num - 1:
                continue
            char_class -= self.char_idx_offset  # remove numbers for the moment
            font_class = int(i / 62)
            attr_value = [0. for i in range(37) ]

            curr_glyph_data = [filename, char_class, font_class, attr_value, fontname]
            curr_font_data.append(curr_glyph_data)
            if (len(curr_font_data) == len(self.chars)):  # collected all glyphs for a font
                dataset.append(curr_font_data)
                curr_font_data = []

        for ii, font in enumerate(dataset):
            if ii % 39 == 10:
                self.val_dataset.append(font)
            elif ii % 39 == 11:
                self.test_dataset.append(font)
            else:
                self.train_dataset.append(font)
        self.val_dataset = self.val_dataset[:self.val_font_num]
        self.test_dataset = self.test_dataset[:self.test_font_num]
        self.train_dataset = self.train_dataset[:self.train_font_num]

        # re-index trainging font for classifcation task
        reindexing_dataset = []
        for ii, font in enumerate(self.train_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.train_dataset = reindexing_dataset

        # re-index val_dataset
        reindexing_dataset = []
        for ii, font in enumerate(self.val_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], self.train_font_num + ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.val_dataset = reindexing_dataset

        # re-index test_dataset
        reindexing_dataset = []
        for ii, font in enumerate(self.test_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], self.train_font_num + self.val_font_num + ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.test_dataset = reindexing_dataset

        ## fontcls2fontname
        self.fontcls2fontname = {}
        for font in self.train_dataset + self.val_dataset + self.test_dataset:
            self.fontcls2fontname[font[0][2]] = font[0][4]

        print('Finished preprocessing the OFL dataset... {} fonts'.format(len(dataset)))
        print('Dataset mode : {}'.format(self.mode))
        print('Trainset : {} fonts'.format(self.train_font_num))
        print('Valset : {} fonts'.format(self.val_font_num))
        print('Testset : {} fonts'.format(self.test_font_num))




    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':

            font_A = self.train_dataset[index]
            label_A = 0.0

            num_glyphs = len(font_A)
            indices = list(range(len(font_A)))
            random.shuffle(indices)  # each __get_item samples two different glyphs
            i = indices[0]  # random sample two glyphs

            glyph_i = font_A[i]

        elif self.mode == 'test' or self.mode == 'val':
            ## index will be flattened font+char
            font_idx = index // self.char_num
            char_idx = index % self.char_num

            if self.mode == 'test':
                font_A = self.test_dataset[font_idx]
            elif self.mode == 'val':
                font_A = self.val_dataset[font_idx]
            glyph_i = font_A[char_idx]
            label_A = 0.0

        filename_i, charclass_i, fontclass_i, attr_i, _ = glyph_i
        image_i = Image.open(os.path.join(self.image_dir, filename_i)).convert('RGB')


        return {"img_i": self.transform(image_i), "charclass_i": torch.LongTensor([charclass_i]),
                "fontclass": torch.LongTensor([fontclass_i]), "attr": torch.FloatTensor(attr_i),
                "label_A": torch.FloatTensor([label_A]),
                }

    def __len__(self):
        if self.mode == 'train':
            """Return the number of images."""
            return self.train_font_num
        elif self.mode == 'test':
            return self.char_num * self.test_font_num
        elif self.mode == 'val':
            return self.char_num * self.val_font_num

class TwoGlyphsPerFont(data.Dataset):
    """Dataset class for the OFL dataset."""
    def __init__(self, dataset_name, image_dir, attr_path,
                transform,  
                mode,
                identical_augmentation=False,
                binary=False,
                char_num=52):
        """Initialize and preprocess the ImageAttr dataset."""
        assert(dataset_name in ['OFL', 'Capitals64'])
        self.dataset_name = dataset_name
        
        self.image_dir = image_dir
        self.attr_path = attr_path

        self.transform = transform
        # self.transform2 = transform2
        self.identical_augmentation = identical_augmentation
        self.mode = mode
        self.binary = binary

        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []


        """
        01234567789abcdefg...xyzABCDE...XYZ
        """
        # self.test_super_unsuper = {}
        # for super_font in range(self.train_font_num+self.val_font_num):
        #     self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)


        assert(self.mode == 'train')
        if dataset_name == 'OFL':
            self.idx2chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.char_num = char_num
            self.char_idx_offset = 10
            self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]
            self.preprocess_OFL()
        elif dataset_name == 'Capitals64':
            self.idx2chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            assert(char_num == 26)
            self.char_num = char_num
            self.char_idx_offset = 0
            self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]
            self.preprocess_Capitals64()

    def preprocess_Capitals64(self):
        self.image_dir = os.path.join(self.image_dir, 'train_split')
        font_names = os.listdir(self.image_dir)
        attr_value = [0. for i in range(37) ] ## NOTE :dummy
        dataset = []
        for ii, font_class in enumerate(font_names):
            # font_path = os.path.join(self.image_dir, font_class)
            curr_font_data = []
            for char_class in range(self.char_num):
                char = self.idx2chars[char_class]
                filename = font_class + '/' + char + '.png'
                curr_glyph_data = [filename, char_class, ii, attr_value, font_class]
                curr_font_data.append(curr_glyph_data)
            dataset.append(curr_font_data)

        self.train_dataset = dataset
        self.train_font_num = len(self.train_dataset)
        assert(self.train_font_num == 7649)
        print('Finished preprocessing the Capitals64 dataset... {} fonts'.format(len(dataset)))
        print('Dataset mode : {}'.format(self.mode))
        print('Trainset : {} fonts'.format(len(self.train_dataset)))


    def preprocess_OFL(self):
        # XXX HARDCODED atm
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        curr_font_data = []
        dataset = []
        for i, line in enumerate(lines):
            filename = line
            fontname = filename.split('/')[0]
            char_class = i % 62
            if char_class < self.char_idx_offset: #XXX  hardcoded excluding nums
                continue
            char_class -= self.char_idx_offset  # remove numbers for the moment
            font_class = int(i / 62)
            attr_value = [0. for i in range(37) ]

            curr_glyph_data = [filename, char_class, font_class, attr_value, fontname]
            curr_font_data.append(curr_glyph_data)
            if (len(curr_font_data) == len(self.chars)):  # collected all glyphs for a font
                dataset.append(curr_font_data)
                curr_font_data = []

        for ii, font in enumerate(dataset):
            if ii % 39 == 10:
                self.val_dataset.append(font)
            elif ii % 39 == 11:
                self.test_dataset.append(font)
            else:
                self.train_dataset.append(font)

        print('Finished preprocessing the OFL dataset... {} fonts'.format(len(dataset)))
        print('Dataset mode : {}'.format(self.mode))
        print('Trainset : {} fonts'.format(len(self.train_dataset)))
        print('Valset : {} fonts'.format(len(self.val_dataset)))
        print('Testset : {} fonts'.format(len(self.test_dataset)))

        self.train_font_num = len(self.train_dataset)
        self.test_font_num = len(self.test_dataset)
        self.val_font_num = len(self.val_dataset)
        # re-index trainging font for classifcation task
        reindexing_dataset = []
        for ii, font in enumerate(self.train_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.train_dataset = reindexing_dataset

        # re-index val_dataset
        reindexing_dataset = []
        for ii, font in enumerate(self.val_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], self.train_font_num + ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.val_dataset = reindexing_dataset

        # re-index test_dataset
        reindexing_dataset = []
        for ii, font in enumerate(self.test_dataset):
            reindex_font = []
            for glyph in font:
                reindex_font.append([glyph[0], glyph[1], self.train_font_num + self.val_font_num + ii, glyph[3], glyph[4]])
            reindexing_dataset.append(reindex_font)
        self.test_dataset = reindexing_dataset

        ## fontcls2fontname
        self.fontcls2fontname = {}
        for font in self.train_dataset + self.val_dataset + self.test_dataset:
            self.fontcls2fontname[font[0][2]] = font[0][4]



    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        font_A = self.train_dataset[index]
        label_A = 0.0

        num_glyphs = len(font_A)
        indices = list(range(len(font_A)))
        random.shuffle(indices)  # each __get_item samples two different glyphs
        i = indices[0]  # random sample two glyphs
        j = indices[1]  # random sample two glyphs

        glyph_i = font_A[i]
        glyph_j = font_A[j]

        filename_i, charclass_i, fontclass_i, attr_i, _ = glyph_i
        filename_j, charclass_j, fontclass_j, attr_j, _ = glyph_j

        assert(fontclass_i == fontclass_j), "fontclass should match"
        assert(charclass_i != charclass_j), "charclass should be different"

        image_i = Image.open(os.path.join(self.image_dir, filename_i)).convert('RGB')
        image_j = Image.open(os.path.join(self.image_dir, filename_j)).convert('RGB')
        if self.identical_augmentation:
            ## https://github.com/pytorch/vision/issues/9#issuecomment-383110707
            seed = random.randint(0,2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            image_i = self.transform(image_i)
            random.seed(seed)
            torch.manual_seed(seed)
            image_j = self.transform(image_j)
        else:
            image_i = self.transform(image_i)
            image_j = self.transform(image_j)

        # image_j = self.transform2(image_i)
        # image_i = self.transform(image_i)

        return {"img_i": image_i , "charclass_i": torch.LongTensor([charclass_i]),
                "img_j": image_j, "charclass_j": torch.LongTensor([charclass_j]),
                "fontclass": torch.LongTensor([fontclass_i]), "attr": torch.FloatTensor(attr_i),
                "label_A": torch.FloatTensor([label_A]),
                }

    def __len__(self):
        """Return the number of images."""
        return self.train_font_num
"""
loader
"""

def get_loader_fontemb(image_dir, attr_path, image_size=256,
               batch_size=16, dataset_name='donovan_embbeding', mode='train', num_workers=8,
               binary=False,
               unsuper_num=968, test_num=100 , char_set='alphabets',
               augmentation=True,
               identical_augmentation=False,
               ):
    """Build and return a data loader."""
    transform = []
    if mode == "train" and augmentation:
        print("Augmentation")
        transform.append(T.RandomResizedCrop(image_size, scale=(0.6,1), ratio=(1,1)
            # interpolation=T.InterpolationMode.BILINEAR
            # interpolation=T.InterpolationMode.BICUBIC
            # interpolation=T.InterpolationMode.NEAREST
            ))
    else:
        print("No Augmentation")
        transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    # transform2 = []
    # transform2.append(T.Resize(image_size))
    # transform2.append(T.ToTensor())
    # transform2.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    # transform2 = T.Compose(transform2)



    if mode in ['test' , 'val', 'train_test']:
        if batch_size == 52 and char_set == 'alphabets':
            pass
        elif batch_size == 10 and char_set == 'numbers':
            pass
        elif batch_size == 62 and char_set == 'alphanumeric':
            pass
        elif batch_size == 26 and char_set == 'capitals':
            pass
        else:
            raise RuntimeError("Test mode not support this batchsize {} with charset {}".format(batch_size, char_set))

    if dataset_name == 'donovan_embedding':
        dataset = TwoGlyphsPerFont_Donovan(image_dir, attr_path, transform, mode,
                binary,
                unsuper_num=unsuper_num, train_num=120, val_num=28)
    elif dataset_name == 'donovan_embedding_per_char':
        dataset = OneGlyphPerFont_Donovan(image_dir, attr_path, transform, mode,
                binary,
                unsuper_num=unsuper_num, train_num=120, val_num=28, char_set=char_set)
    elif dataset_name == 'ofl':
        dataset = TwoGlyphsPerFont('OFL', image_dir, attr_path, transform,  mode,
                identical_augmentation,
                binary)
    elif dataset_name == 'ofl_per_char':
        dataset = OneGlyphPerFont('OFL', image_dir, attr_path, transform, mode,
                binary,
                char_set=char_set)
    elif dataset_name == 'Capitals64':
        dataset = TwoGlyphsPerFont('Capitals64', image_dir, attr_path, transform,  mode,
                identical_augmentation,
                binary,
                char_num=26)
    elif dataset_name == 'Capitals64_per_char':
        dataset = OneGlyphPerFont('Capitals64', image_dir, attr_path, transform, mode,
                binary,
                char_set=char_set)
    else:
        raise NotImplementedError("Need to implement this data : {}".format(dataset_name))

    shuffle = (mode == 'train')
    # shuffle = False
    data_loader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=shuffle, 
                                  num_workers=num_workers)

    return data_loader
