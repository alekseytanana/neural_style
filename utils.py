import os
import IPython
from decimal import Decimal



class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def log(message, verbose=True):
    if not verbose:
        return
    print(message)


def get_style_image_paths(style_image_input):
    style_image_list = []
    for path in style_image_input:
        if os.path.isdir(path):
            images = (os.path.join(path, file) for file in os.listdir(path) 
                      if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".tiff"])
            style_image_list.extend(images)
        else:
            style_image_list.append(path)
    return style_image_list


def maybe_update(net, t, update_iter, num_iterations, loss):
    if update_iter != None and t % update_iter == 0:
        IPython.display.clear_output()
        print('Iteration %d/%d: '%(t, num_iterations))
        if net.content_weight > 0:
            print('  Content loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.content_losses]))
        print('  Style loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.style_losses if module.strength > 0]))
        print('  Histogram loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.hist_losses if module.strength > 0]))
        if net.tv_weight > 0:
            print('  TV loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.tv_losses]))
        print('  Total loss = %.2e' % Decimal(loss.item()))

        
def maybe_save_preview(img, t, save_iter, num_iterations, output_path):
    should_save = save_iter > 0 and t % save_iter == 0
    if not should_save:
        return
    output_filename, file_extension = os.path.splitext(output_path)
    #output_filename = output_filename.replace('results', 'results/preview')
    filename = '%s_%04d%s' % (output_filename, t, file_extension)
    save(deprocess(img), filename)

    