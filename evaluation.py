import torch
from io import StringIO
from contextlib import redirect_stdout

def evaluate_ret_acc(fontemb_net, F, test_dataloader, verbose=False):
    # load
    if verbose:
        fontemb_net.load_state_dict(torch.load(F))
    else:
        with redirect_stdout(StringIO()) as f:
            fontemb_net.load_state_dict(torch.load(F))
    fontemb_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # infer
    import time
    t = time.time()

    c_list = []
    char_num = test_dataloader.dataset.char_num

    feat_per_char = []
    ret_dict = {}
    fontemb_net.eval()

    features = []
    with torch.no_grad():
        for ii, batch_test in enumerate(test_dataloader):
            img_i = batch_test['img_i'].to(device)
            feat_i, _  = fontemb_net(img_i)
            features.append(feat_i)

    features_per_font = torch.stack(features) ##  torch.Size([100, 52, 512])
    feat_per_char = features_per_font.permute(1,0,2) #  torch.Size([52, 100, 512])

    # ret_accuracy_upper, ret_per_char_upper = retrieval_evaluation(feat_per_char, test_dataloader, init_char=26, verbose=False)
    ret_accuracy, ret_per_char = retrieval_evaluation(feat_per_char, test_dataloader, init_char=0, verbose=False)

    # return ret_accuracy, ret_per_char, ret_accuracy_upper, ret_per_char_upper
    return ret_accuracy, ret_per_char ##, ret_accuracy_upper, ret_per_char_upper

def retrieval_evaluation(feat_per_char, test_dataloader, init_char = 26, verbose=True):  ## retrieve capitals only
    c_list = []
    ret_dict = {}
    char_num = test_dataloader.dataset.char_num
    testchars =  test_dataloader.dataset.idx2chars
    num_font = feat_per_char[0].shape[0]
    print("# of fonts: {}".format(num_font))

    with torch.no_grad():
        for ii in range(init_char, len(testchars)):
            for jj in range(ii+1, len(testchars)):
                # print(ii,jj)

                feat_i = feat_per_char[ii]
                feat_j = feat_per_char[jj]
                cdist = torch.cdist(feat_i, feat_j, 2)

                ## A -> B
                retrieval_prediction = list(torch.argmin(cdist, dim=1).cpu().numpy())
                c = 0
                for i in range(num_font):
                    if i == retrieval_prediction[i]:
                        c += 1
                # print("{}/{} correct".format(c, num_font))
                c_list.append(c)
                ret_dict["{}->{}".format(testchars[ii], testchars[jj])] = c
                # B -> A
                retrieval_prediction = list(torch.argmin(cdist, dim=0).cpu().numpy())
                c = 0
                for i in range(num_font):
                    if i == retrieval_prediction[i]:
                        c += 1
                # print("{}/{} correct".format(c, num_font))
                c_list.append(c)
                ret_dict["{}->{}".format(testchars[jj],testchars[ii])] = c
    ret_accuracy = sum(c_list)/len(c_list)/num_font*100
    # print(ret_dict)

    ret_per_char = {}
    for ii in range(init_char, len(testchars)):
        c = testchars[ii]
        c_ret = [v for k, v in ret_dict.items() if k.startswith(c)]
        c_ret = sum(c_ret)/len(c_ret)/num_font*100
        ret_per_char[c] = c_ret
    ret_per_char = {e[0]+"_ret_accuracy":e[1] for e in sorted(ret_per_char.items(), key=lambda x: x[1])}
    if verbose:
        for k,v in ret_per_char.items():
            print( "{0:>20s}".format(k) + "  {0:6f}".format(v))
        print("Retrieval accuracy : higher the better")
    return ret_accuracy, ret_per_char


def QR_evaluation(fontemb_net, test_dataloader, criterion_eval):
    device = next(fontemb_net.parameters()).device # NOTE: assume model in one gpu
    """
    Test Q<->R embedding loss
    """
    test_dataloader.dataset.change_infer_char(42)
    batch_test = next(iter(test_dataloader))
    img_i = batch_test['img_i'].to(device)
    test_dataloader.dataset.change_infer_char(43)
    batch_test = next(iter(test_dataloader))
    img_j = batch_test['img_i'].to(device)


    ## model forward & backward
    _, output_i = fontemb_net(img_i)
    _, output_j = fontemb_net(img_j)

    z_i = output_i[0]
    z_j = output_j[0]  # NOTE: 0 index of head is always donovan

    loss_dict = criterion_eval(z_i, z_j)
    return loss_dict

def attribute_evaluation(attr_per_char, attr_gt, test_dataloader, criterion_attr_eval):
    attribute_error_char = {}
    loss_all = []

    # for i in range(26):
    for attr_i in attr_per_char:
        ## attr_i : [28, 37]
        ## attr_gt :  [28, 37]
        loss_i = criterion_attr_eval(attr_i, attr_gt)
        loss_all.append(loss_i)

    loss_all = torch.stack(loss_all)   ## 52 chars 28 fonts 37 attributes
    error_per_attr = loss_all.mean(0).mean(0)    ###  error per attributes : dim 37
    error_per_chars = loss_all.mean(1).mean(1)    ###  error per attributes : dim 52
    error_attr = error_per_attr.mean()
    # Visualzie log
    attr_error_dict = {test_dataloader.dataset.idx2attr[i] : err.item()
            for i , err in enumerate(error_per_attr)}
    attr_error_dict = {e[0]:e[1] for e in sorted(attr_error_dict.items(), key=lambda x: x[1])}
    char_error_dict = {test_dataloader.dataset.idx2chars[i] + "_attr_error" : err.item()
            for i , err in enumerate(error_per_chars)}
    char_error_dict = {e[0]:e[1] for e in sorted(char_error_dict.items(), key=lambda x: x[1])}
    lines = []
    for k, v in char_error_dict.items():
        lines.append("{0:>20s}".format(k) + "  {0:6f}".format(v))

    for i, (k, v) in enumerate(attr_error_dict.items()):
        ll = "{0:>20s}".format(k) + "  {0:6f}".format(v)
        lines[i] += ll
    message = "Attribute error : lower the better\n"
    for ll in lines:
        message += ll
        message += "\n"
    return error_attr, attr_error_dict, char_error_dict, message 


def L1_gen_evaluation(fontemb_net, fontdec_net, val_dataloader, verbose=False):
    num_chars = val_dataloader.dataset.char_num
    idx2chars = val_dataloader.dataset.idx2chars
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_pixel = torch.nn.L1Loss().to(device)

    epoch_error = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            charclass = batch['charclass_i'].to(device)
            image = batch['img_i'].to(device)
            
            feature, _ = fontemb_net(image)

            L1_error_per_char = {}
            for ii in range(num_chars):

                repeated_feature = feature[ii].repeat(52,1)
                one_hot = torch.nn.functional.one_hot(charclass, num_classes=num_chars).to(device).squeeze(1)
                feat_char_cat = torch.cat([repeated_feature, one_hot], dim=1)

                out = fontdec_net(feat_char_cat)    
                L1_error = criterion_pixel(out, image)

                L1_error_per_char[idx2chars[ii]] = L1_error.item()
            
            # L1_error_per_char_all.append(L1_error_per_char)
            mean_L1_per_font = sum([v for _,v in L1_error_per_char.items()]) / num_chars
            if verbose:
                print("{}/{} : {}".format(i, len(val_dataloader), mean_L1_per_font))
            
            epoch_error.append(mean_L1_per_font)
    L1_error = sum(epoch_error) / len(epoch_error)
    return L1_error
