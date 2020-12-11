import torch
from model import *
from utils import *



_stylenet_config_keys_ = [ 'type',
    'size', 'content_image', 'style_images', 
    'style_image', 'content_masks', 'style_blend_weights', 
    'num_octaves', 'style_scale', 'num_iterations', 
    'octave_ratio', 'original_colors',
    'tv_weight', 'content_weight', 'style_weight', 'hist_weight', 
    'style_stat', 'normalize_gradients'
]


def optimize(stylenet, 
             img, 
             num_iterations, 
             update_iter=250, 
             display_preview=False, 
             save_preview=False, 
             save_preview_path=None,
             clear_output=True):

    def iterate_optimizer():
        t[0] += 1

        optimizer.zero_grad()
        stylenet(img)
        loss = stylenet.get_loss()
        loss.backward()
        
        if update_iter is not None:
            maybe_update(stylenet, t[0], update_iter, num_iterations, loss)
            if t[0] % update_iter == 0 or t[0]==0:
                display(deprocess(img))
        if save_preview:
            maybe_save_preview(img, t[0], update_iter, num_iterations, save_preview_path)

        return loss
    
    img = nn.Parameter(img.type(stylenet.dtype))
    optimizer, loopVal = setup_optimizer(img, stylenet.params, num_iterations, verbose=False)
    t = [0]
    while t[0] <= loopVal:
        optimizer.step(iterate_optimizer)
    
    if update_iter is not None and clear_output:
        IPython.display.clear_output()   
    
    return img


def style_transfer(stylenet, config, img=None, verbose=False):
    cfg = EasyDict(config)

    cfg.size = cfg.size if 'size' in cfg else 512
    cfg.content_image = cfg.content_image if 'content_image' in cfg else None
    cfg.style_images = cfg.style_image if ('style_image' in cfg and 'style_images' not in cfg) else cfg.style_images
    cfg.style_images = cfg.style_images if 'style_images' in cfg else None 
    cfg.style_images = cfg.style_images if isinstance(cfg.style_images, list) else [cfg.style_images]
    cfg.content_masks = cfg.content_masks if 'content_masks' in cfg else None 
    cfg.style_blend_weights = cfg.style_blend_weights if 'style_blend_weights' in cfg else None
    cfg.num_iterations = cfg.num_iterations if 'num_iterations' in cfg else 1000
    cfg.num_iterations = cfg.num_iterations if isinstance(cfg.num_iterations, list) else [cfg.num_iterations]    
    cfg.num_octaves = int(cfg.num_octaves) if 'num_octaves' in cfg else len(cfg.num_iterations)
    cfg.num_iterations = cfg.num_iterations * cfg.num_octaves if len(cfg.num_iterations) == 1 else cfg.num_iterations
    cfg.style_scale = cfg.style_scale if 'style_scale' in cfg else 1.0
    cfg.style_scale = cfg.style_scale if isinstance(cfg.style_scale, list) else [cfg.style_scale]
    cfg.style_scale = cfg.style_scale * cfg.num_octaves if len(cfg.style_scale) == 1 else cfg.style_scale
    cfg.octave_ratio = float(cfg.octave_ratio) if 'octave_ratio' in cfg else 1.0
    cfg.original_colors = cfg.original_colors if 'original_colors' in cfg else False
    cfg.tv_weight = cfg.tv_weight if 'tv_weight' in cfg else default_tv_weight
    cfg.content_weight = cfg.content_weight if 'content_weight' in cfg else default_content_weight
    cfg.style_weight = cfg.style_weight if 'style_weight' in cfg else default_style_weight
    cfg.hist_weight = cfg.hist_weight if 'hist_weight' in cfg else default_hist_weight
    cfg.style_stat = cfg.style_stat if 'style_stat' in cfg else default_style_stat
    cfg.normalize_gradients = cfg.normalize_gradients if 'normalize_gradients' in cfg else default_normalize_gradients
    
    # checks
    extraneous_keys = [k for k in cfg.keys() if k not in _stylenet_config_keys_]
    assert len(extraneous_keys) == 0, \
        'Following config keys are not recognized: %s' % ', '.join(extraneous_keys)
    warn(cfg.num_octaves > 1 and cfg.octave_ratio == 1.0, \
         'Multi-resolution (num_octaves>1) but octave_ratio is 1.0', verbose)
    assert 'style_image' in cfg or 'style_images' in cfg, \
        'Error: must specify at least one style image'
    assert not (cfg.content_weight > 0 and cfg.content_image is None), \
        'Error: if no content image provided, content_weight must be 0'
    assert cfg.style_blend_weights is None or len(cfg.style_blend_weights) == len(cfg.style_images), \
        'Error: number of style_blend_weights elements must match number of styles'
    assert cfg.num_octaves == len(cfg.num_iterations) == len(cfg.style_scale), \
        'Error: num_octaves does not match length of num_iterations or style_scale lists'

    # load original content image
    if cfg.content_image is None:
        assert isinstance(cfg.size, tuple), \
            'Error: If no content image provided, config.size must be a full tuple (width, height)'
        content_image_orig = deprocess(random_tensor(cfg.size[0], cfg.size[1]))
    else:
        content_image_orig = load_image(cfg.content_image, cfg.size)
        cfg.size = get_size(content_image_orig)  

    # load original style images, and save aspect ratios
    max_size = max(cfg.size) if isinstance(cfg.size, tuple) else cfg.size
    style_images_orig = [load_image(image, int(max_size * max(cfg.style_scale))) 
                                    for image in cfg.style_images]
    style_images_aspect = [get_aspect_ratio(image) for image in style_images_orig]

    # load original content masks
    if cfg.content_masks is not None:
        if isinstance(cfg.content_masks, np.ndarray):
            num_masks = cfg.content_masks.shape[-1]
            content_masks_orig = [mask_to_image(cfg.content_masks[:,:,m]) 
                                  for m in range(num_masks)]
        else:
            cfg.content_masks = cfg.content_masks if isinstance(cfg.content_masks, list) else [cfg.content_masks]
            content_masks_orig = [load_image(mask, cfg.size) 
                                  for mask in cfg.content_masks]
        assert len(content_masks_orig) == len(cfg.style_images), \
            'Error: Number of content masks (%d) != number of styles (%d)' % (len(content_masks_orig), len(cfg.style_images))
    else:
        content_masks_orig = None

    # load initial input image
    if img is None:
        img = random_tensor_like(content_image_orig)
    else:
        img = preprocess(img)
    
    # calculate image sizes
    if isinstance(cfg.size, tuple):
        image_sizes = [(int(cfg.size[0] * (cfg.octave_ratio ** -s)), 
                        int(cfg.size[1] * (cfg.octave_ratio ** -s))) 
                       for s in reversed(range(cfg.num_octaves))]
    else:
        aspect = get_aspect_ratio(content_image_orig)
        image_sizes = [(int(cfg.size * (cfg.octave_ratio ** -s) * aspect), 
                        int(cfg.size * (cfg.octave_ratio ** -s))) 
                       for s in reversed(range(cfg.num_octaves))]
    
    # push stylenet hyperparameters
    stylenet.save_parameters()
    stylenet.set_tv_weight(cfg.tv_weight)
    stylenet.set_content_weight(cfg.content_weight)
    stylenet.set_style_weight(cfg.style_weight)
    stylenet.set_hist_weight(cfg.hist_weight)
    stylenet.set_style_statistic(cfg.style_stat)
    stylenet.set_normalize_gradients(cfg.normalize_gradients)

    # go through each octave
    for image_size, num_iterations, style_scale in zip(image_sizes, cfg.num_iterations, cfg.style_scale):

        # rescale main image
        img = resize_tensor(img, image_size)        
        
        # reload content, style, and mask images at scale
        content_image = resize(content_image_orig, image_size)
        style_images = [resize(image, (int(max(image_size) * style_scale), 
                                       int(max(image_size) * style_scale / aspect))) 
                        for image, aspect in zip(style_images_orig, style_images_aspect)]
        
        # produce masks at current octave size
        if content_masks_orig is not None:
            content_masks = [resize(mask, image_size) 
                             for mask in content_masks_orig]
        else:
            content_masks = None

        # capture the style and content images
        stylenet.capture(content_image, 
                         style_images, 
                         cfg.style_blend_weights, 
                         content_masks)

        # optimize
        img = optimize(stylenet, 
                       img, 
                       num_iterations=num_iterations,  
                       update_iter=100, 
                       display_preview=True,
                       save_preview=False,
                       save_preview_path='results/preview/preview.png')
    
    # tensor to PIL
    img = deprocess(img)
    
    # change back to original colors of content image
    if cfg.original_colors and cfg.content_image is not None:
        img = original_colors(content_image, img)
    
    # pop stylenet hyperparameters
    stylenet.restore_parameters()
        
    return img
