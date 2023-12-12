import numpy as np
import PIL
from skimage import color
from scipy import stats, ndimage
from scipy.signal import correlate
from skimage.transform import resize, rotate
import skimage.measure
from skimage.filters import threshold_otsu


PIL.Image.MAX_IMAGE_PIXELS = 10000000000 


### Mean RGB, Mean HSV, Mean LAB
def mean_channels(img, circ=False):
    # returns 1 value for grayscale and 3 values for color channels
    return np.mean(img, axis=(0,1))
    
### STD of RGB, HSV, LAB; STD(L)="RMSContrast"
def std_channels(img):
    # returns 1 value for grayscale and 3 values for color channels
    return np.std(img, axis=(0,1))


def circ_stats(img_hsv):
    hue = img_hsv[:,:,0].astype("float")
    circ_mean = stats.circmean(hue, high=1, low=0)
    circ_std = stats.circstd(hue, high=1, low=0)
    return circ_mean, circ_std
        
    
### color entropy, shannon_entropy Gray, 
def shannonentropy_channels(img):
    ### grayscale 2 dim image
    if img.ndim == 2:
        return skimage.measure.shannon_entropy(img)
    if img.ndim == 3:
        chan_0 = skimage.measure.shannon_entropy(img[:,:,0])
        chan_1 = skimage.measure.shannon_entropy(img[:,:,1])
        chan_2 = skimage.measure.shannon_entropy(img[:,:,2])
        return chan_0, chan_1, chan_2

def aspect_ratio(img_RGB):
    return img_RGB.shape[1] / img_RGB.shape[0]

def image_size(img_RGB):
    return img_RGB.shape[1] + img_RGB.shape[0] 

################################################################################
######################### slope and sigma ##################################
################################################################################


def sfsl_get_radius(_input, img_input = True):
    shape = _input.shape
    ######%get cartesian coordinates for shifted image (center in the middle)
    grid_x = np.arange(1, shape[1] + 1) - shape[1]/2 - 1
    grid_y = np.arange(1, shape[0] + 1) - shape[0]/2 - 1
    #np.meshgrid(grid_x,grid_y)[1]  
    x_idx, y_idx = np.meshgrid(grid_x,grid_y)
    
    return np.round( np.sqrt(x_idx**2 + y_idx**2) )


def sfsl_calc_radial_average(img):
    #### get matrix of radii for fftshifted image
    rad = sfsl_get_radius(img);
    #### get sorted list of radius entries
    r_list = np.unique(rad)
    ###preallocate memory for result
    avg = np.zeros(len(r_list)); 
    for r in range(len(r_list)):
        avg[r] = np.mean(img[ rad  == r_list[r]  ]) 
    return avg, r_list


def tff_image(img):    
    ### fastfuriertransform
    f = np.fft.fft2(img, axes=(0, 1))
    ### power and ffshift
    f_shift = np.fft.fftshift(np.abs(f)**2) #,  axes=(0, 1))

    return  f_shift


def fourier_sigma(img):
    f_pow = tff_image( img )
    avg, r_list = sfsl_calc_radial_average(f_pow)
    l_lim = 10 ### cutoff artifacts
    u_lim =  256 ### cutoff artifacts
    y = np.log10( avg[ l_lim : u_lim] )
    x = np.log10( r_list[ l_lim : u_lim] )
    slope, intercept, r_value, p_value, std_err = stats.linregress(x , y)
    
    line_points = x*slope + intercept
    sigma = np.mean((line_points - y)**2)

    return sigma , slope
        
################################################################################
################################# Edge Entropy ########################################
################################################################################

def create_gabor(size, theta=0, octave=3):
    amplitude = 1.0
    phase = np.pi/2.0
    frequency = 0.5**octave # 0.5**(octave+0.5)
    hrsf = 4 # half response spatial frequency bandwith
    sigma = 1/(np.pi*frequency) * np.sqrt(np.log(2)/2) * (2.0**hrsf+1)/(2.0**hrsf-1)
    valsy = np.linspace(-size//2+1, size//2, size)
    valsx = np.linspace(-size//2+1, size//2, size)
    xgr,ygr = np.meshgrid(valsx, valsy);

    omega = 2*np.pi*frequency
    gaussian = np.exp(-(xgr*xgr + ygr*ygr)/(2*sigma*sigma))
    slant = xgr*(omega*np.sin(theta)) + ygr*(omega*np.cos(theta))

    gabor = np.round(gaussian, decimals=4) * amplitude*np.cos(slant + phase);
    # e^(-(x^2+y^2)/(2*1.699^2)) *cos(pi/4*(x*sin(2)+y*cos(2)) + pi/2)

    return np.round(gabor, decimals=4)

def create_filterbank(flt_size=31 , num_filters=24):
    flt_raw = np.zeros([num_filters, flt_size, flt_size])
    BINS_VEC = np.linspace(0, 2*np.pi, num_filters+1)[:-1]
    for i in range(num_filters):
        flt_raw[i,:,:] = create_gabor(flt_size, theta=BINS_VEC[i], octave=3)
        #print(i, flt_size, BINS_VEC[i])
    return flt_raw


def run_filterbank(flt_raw, img, num_filters=24):
    (h, w) = img.shape
    num_filters = flt_raw.shape[0]
    image_flt = np.zeros((num_filters,h,w))

    for i in range(num_filters):
        image_flt[i,:,:] = ndimage.convolve(img, flt_raw[i,:,:])

    resp_bin = np.argmax(image_flt, axis=0)
    resp_val = np.max(image_flt, axis=0)
    
    return resp_bin, resp_val


def edge_density(resp_val):
    normalize_fac = float(resp_val.shape[0] * resp_val.shape[1])
    edge_d = np.sum(resp_val)/normalize_fac
    return edge_d


def do_counting(resp_val, resp_bin, CIRC_BINS=48, GABOR_BINS=24, MAX_DIAGONAL = 500):
    """creates histogram (distance, relative orientation in image, relative gradient)"""
    
    h, w = resp_val.shape;
   
    # cutoff minor filter responses
    cutoff = np.sort(resp_val.flatten())[-10000] # get 10000th highest response for cutting of beneath
    resp_val[resp_val<cutoff] = 0
    ey, ex = resp_val.nonzero()

    # lookup tables to speed up calculations
    edge_dims = resp_val.shape
    xx, yy = np.meshgrid(np.linspace(-edge_dims[1],edge_dims[1],2*edge_dims[1]+1), np.linspace(-edge_dims[0],edge_dims[0],2*edge_dims[0]+1))
    dist = np.sqrt(xx**2+yy**2)

    orientations = resp_bin[ey,ex]
    counts = np.zeros([500, CIRC_BINS, GABOR_BINS])

    print ("Counting", 'image name', resp_val.shape, "comparing", ex.size)
    for cp in range(ex.size):

        orientations_rel = orientations - orientations[cp]
        orientations_rel = np.mod(orientations_rel + GABOR_BINS, GABOR_BINS)

        distance_rel = np.round(dist[(ey-ey[cp])+edge_dims[0], (ex-ex[cp])+edge_dims[1]]).astype("uint32")
        distance_rel[distance_rel>=MAX_DIAGONAL] = MAX_DIAGONAL-1

        direction = np.round(np.arctan2(ey-ey[cp], ex-ex[cp]) / (2.0*np.pi)*CIRC_BINS + (orientations[cp]/float(GABOR_BINS)*CIRC_BINS)).astype("uint32")
        direction = np.mod(direction+CIRC_BINS, CIRC_BINS)
        np.add.at(counts, tuple([distance_rel, direction, orientations_rel]), resp_val[ey,ex] * resp_val[ey[cp],ex[cp]])

    return counts, resp_val


def entropy(a):
    if np.sum(a)!=1.0 and np.sum(a)>0:
        a = a / np.sum(a)
    v = a>0.0
    return -np.sum(a[v] * np.log2(a[v]))


def do_statistics(counts, GABOR_BINS=24):
    # normalize by sum
    counts_sum = np.sum(counts, axis=2) + 0.00001
    normalized_counts = counts / (counts_sum[:,:,np.newaxis])
    d,a,_ = normalized_counts.shape
    shannon_nan = np.zeros((d,a))
    for di in range(d):
      for ai in range(a):
        if counts_sum[di,ai]>1:  ## ignore bins without pixels
            shannon_nan[di,ai] = entropy(normalized_counts[di,ai,:])
        else:
            shannon_nan[di,ai] = np.nan
    return shannon_nan

         
def edge_resize (img_gray_np, max_pixels = 300*400):
    if max_pixels != None:
        img_gray_PIL = PIL.Image.fromarray(img_gray_np)
        s0,s1 = img_gray_PIL.size
        a = np.sqrt(max_pixels / float(s0*s1))
        img_gray_PIL_rez = img_gray_PIL.resize((int(s0*a),int(s1*a)), PIL.Image.LANCZOS)
        img_gray_np = np.asarray(img_gray_PIL_rez, dtype='float')
    return img_gray_np

# def do_second_order (img):
#     flt_raw = create_filterbank()
#     img = edge_resize (img)
#     resp_bin, resp_val = run_filterbank(flt_raw, img)
#     counts = do_counting(resp_val, resp_bin)
#     shannon_nan = do_statistics(counts)
        
#     RANGES = [(20,240)]
#     results = []
#     for r in RANGES:
#         # with warnings.catch_warnings():
#         #     warnings.simplefilter("ignore", category=RuntimeWarning)
#         results.append(np.nanmean(np.nanmean(shannon_nan, axis=1)[r[0]:r[1]]))
#     return results
    
# def do_first_order(img, resp_val, resp_bin, GABOR_BINS=24):
#     flt_raw = create_filterbank()
#     img = edge_resize (img)
#     resp_bin, resp_val = run_filterbank(flt_raw, img)
#     first_order_bin = np.zeros(GABOR_BINS)
#     for b in range(GABOR_BINS):
#         first_order_bin[b] = np.sum(resp_val[img.resp_bin==b])
#     first_order = entropy(first_order_bin)
#     return first_order


def do_first_and_second_order_entropy_and_edge_density (img, GABOR_BINS=24):
    flt_raw = create_filterbank()
    img = edge_resize (img)
    resp_bin, resp_val = run_filterbank(flt_raw, img)
    
    ### edge density
    edge_d = edge_density(resp_val)
    
    ### do_counting must run before first_order_entropy but after edge density because it modifies resp_val!!! 
    counts, resp_val = do_counting(resp_val, resp_bin)

    ### first order entropy
    first_order_bin = np.zeros(GABOR_BINS)
    for b in range(GABOR_BINS):
        first_order_bin[b] = np.sum(resp_val[resp_bin==b])
    first_order = entropy(first_order_bin)
    ###second order entropy
    shannon_nan = do_statistics(counts)
    second_order = np.nanmean(np.nanmean(shannon_nan, axis=1)[20:240])
    return first_order, second_order, edge_d


################################# LAB Color rotation ################################

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def rotate_image_in_LAB_colorspace(img_lab, degree):
    degree_in_pi = degree/(180)
    
    ### get polar coordinates for each pixel
    rho, phi = cart2pol(img_lab[:,:,1], img_lab[:,:,2])
  
    ### change angle
    phi = phi+  degree_in_pi * np.pi
    
    ### convert back to polar coordinates
    x, y = pol2cart(rho, phi)
    
    ## assign to image, ceeping original luminance
    img_lab[:,:,1] = x
    img_lab[:,:,2] = y
    
    # convert to RGB
    img_RGB_rotated = color.lab2rgb(img_lab)
    
    return img_RGB_rotated


def move_image_in_LAB_colorspace(img_lab, degree):
    degree_in_pi = degree/(180)
    
    ### get polar coordinates for each pixel
    
    rho, phi = cart2pol(img_lab[:,:,1], img_lab[:,:,2])
  
    
    phi = degree_in_pi * np.pi
    
    x, y = pol2cart(rho, phi)
    
    img_lab[:,:,1] = x
    img_lab[:,:,2] = y
    
    
    img_RGB_rot = color.lab2rgb(img_lab)
    
    return img_RGB_rot


##################################################################################
################################ CNN Measures #####################################
##################################################################################



def resize_and_add_ImageNet_mean(img):
    ### resize img to desired dimension
    img = resize(img, [512,512], order=1)  ### Not the same "resize" function as old code, leads to very small differences
    ### normalize with image_net mean
    img = img  - np.array([104.00698793 , 116.66876762 , 122.67891434])
    ### add new additional axis and return
    return img


def conv2d(input_img, kernel, bias):
    
    input_img = input_img[:,:,(2,1,0)].astype(np.float32)  ## Caffe Net used different channel orders
    
    input_img = resize_and_add_ImageNet_mean(input_img )
    
    # Get input data dimensions
    in_height, in_width, in_channels = input_img.shape

    # Get kernel dimensions
    k_height, k_width, in_channels, out_channels = kernel.shape

    # Calculate output dimensions
    out_height = int(np.ceil(float(in_height - k_height + 1) / float(4)))
    out_width = int(np.ceil(float(in_width - k_width + 1) / float(4)))

    # Allocate output data
    output_data = np.zeros((out_height, out_width, out_channels))

    # Convolve each input channel with its corresponding kernel and sum the results
    for j in range(out_channels):
        for i in range(in_channels):
            output_data[:, :, j] += correlate(
                input_img[:, :, i],
                kernel[:, :, i, j],
                mode='valid'
            )[::4, ::4]

        # Add bias to the output
        output_data[:, :, j] += bias[j]
        
    ## relu activation function
    output_data[output_data < 0] = 0
    
    ### swap axis to order: filters, dim1_filters, dim2_filters (96,126,126)
    output_data = np.swapaxes(output_data,2,0)
    output_data = np.swapaxes(output_data,1,2)
        
    return output_data

###################### Variances ####################################################


def get_CNN_Variance(normalized_max_pool_map, kind):
    result = 0
    if kind == 'sparseness':
        result =  np.var( normalized_max_pool_map)
    elif kind == 'variability':
        result =  np.median(np.var(normalized_max_pool_map , axis=(0,1)))
    else:
        raise ValueError("Wrong input for kind of CNN_Variance. Use sparseness or variability")
    return result


def max_pooling (resp, patches ):
    (i_filters, ih, iw) = resp.shape
    max_pool_map = np.zeros((patches,patches,i_filters))
    patch_h = ih/float(patches)
    patch_w = iw/float(patches)

    for h in range(patches):
        for w in range(patches):
            ph = h*patch_h
            pw = w*patch_w
            patch_val = resp[:,int(ph):int(ph+patch_h), int(pw):int(pw+patch_w)]

            for b in range(i_filters):
                max_pool_map[h,w,b] = np.max(patch_val[b])

    max_pool_map_sum = np.sum(max_pool_map, axis=2)
    normalized_max_pool_map = max_pool_map / max_pool_map_sum[:,:,np.newaxis]

    
    return max_pool_map, normalized_max_pool_map

################### Self-Similarity ################################

def get_selfsimilarity(histogram_ground, histogram_level):
    ph, pw, n = histogram_level.shape
    hiks = []
    for ih in range(ph):
        for iw in range(pw):
            hiks.append( np.sum(np.minimum( histogram_ground, histogram_level[ih,iw])) )
    sesim = np.median(hiks)
    return sesim


################### CNN Symmetry ################################


def get_differences(max_pooling_map_orig, max_pooling_map_flip):
    assert(max_pooling_map_orig.shape == max_pooling_map_flip.shape)
    sum_abs = np.sum(np.abs(max_pooling_map_orig - max_pooling_map_flip))
    sum_max = np.sum(np.maximum(max_pooling_map_orig, max_pooling_map_flip))
    return 1.0 - sum_abs / sum_max


def get_symmetry(input_img, kernel, bias):
    
    ### get max pooling map for orig. image
    resp_orig = conv2d(input_img, kernel, bias)
    max_pooling_map_orig, _ = max_pooling (resp_orig, patches=17)
    
    ### get max pooling map for left-right fliped image
    img_lr = np.fliplr(input_img)
    resp_lr = conv2d(img_lr, kernel, bias)
    max_pooling_map_lr, _ = max_pooling (resp_lr, patches=17)
    sym_lr = get_differences(max_pooling_map_orig, max_pooling_map_lr)
    
    ### get max pooling map for up-down fliped image
    img_ud = np.flipud(input_img)
    resp_ud = conv2d(img_ud, kernel, bias)
    max_pooling_map_ud, _ = max_pooling (resp_ud, patches=17)
    sym_ud = get_differences(max_pooling_map_orig, max_pooling_map_ud)

    ### get max pooling map for up-down and left-right fliped image
    img_lrud = np.fliplr(np.flipud(input_img))
    resp_lrud = conv2d(img_lrud, kernel, bias)
    max_pooling_map_lrud, _ = max_pooling (resp_lrud, patches=17)
    sym_lrud = get_differences(max_pooling_map_orig, max_pooling_map_lrud)
    
    return sym_lr, sym_ud, sym_lrud



################################################################################
################################# PHOG ########################################
################################################################################





def resize_img(img, re): 
    if re>1:
        s = img.shape
        b = s[0]*s[1]
        d = np.sqrt(re/b)
        s_new = np.round([s[0]*d, s[1]*d]).astype(np.int32)
        img = resize(img, s_new, order=3, anti_aliasing=True) *255
        return img.astype(np.uint8)
    
    else:
        return img
    

def absmaxND(a, axis=2):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)




def maxGradient_fast(Img):

    gradY_t, gradX_t = np.gradient(Img, axis = [0,1], edge_order=1)
    
    gradientX = absmaxND(gradX_t)
    gradientY = absmaxND(gradY_t)

    return gradientX, gradientY



def computeDescriptor(GradientValue, GradientAngle, bins, angle, levels, section, is_global=False):
    descriptor = []

    intervalSize = angle / bins
    halfIntervalSize = (angle / bins) / 2
    
    # Level 0
    ind = ((GradientAngle >= angle - halfIntervalSize) | (GradientAngle < halfIntervalSize))
    descriptor.append(np.sum(GradientValue[ind]))

    for b in range(1, bins):
        ind = ((GradientAngle >= (b * intervalSize) - halfIntervalSize) & (GradientAngle < ((b + 1) * intervalSize) - halfIntervalSize))
        descriptor.append(np.sum(GradientValue[ind]))

    ### local normaliszation for global
    if is_global:
        descriptor = normalizeDescriptor(descriptor, bins)

    # Other levels
    for l in range(1, levels + 1):
        cellSizeX = GradientAngle.shape[1] / (section ** l)
        cellSizeY = GradientAngle.shape[0] / (section ** l)
        
        if cellSizeX < 1 or cellSizeY < 1:
            raise ValueError("Cell size is less than 1. Adjust the number of levels.")

        for j in range(1, section ** l + 1):

            leftX = 1 + np.round((j - 1) * cellSizeX).astype(np.int64)
            rightX = np.round(j * cellSizeX).astype(np.int64)
            
            for i in range(1, section ** l + 1):

                topY = 1 + np.round((i - 1) * cellSizeY).astype(np.int64)
                bottomY = np.round(i * cellSizeY).astype(np.int64)

                GradientValueCell = GradientValue[topY - 1:bottomY, leftX - 1:rightX]
                GradientAngleCell = GradientAngle[topY - 1:bottomY, leftX - 1:rightX]

                ind = ((GradientAngleCell >= angle - halfIntervalSize) | (GradientAngleCell < halfIntervalSize))
                local_descriptor = [np.sum(GradientValueCell[ind])]

                for b in range(1, bins):
                    ind = ((GradientAngleCell >= (b * intervalSize) - halfIntervalSize) & (GradientAngleCell < ((b + 1) * intervalSize) - halfIntervalSize))
                    local_descriptor.append(np.sum(GradientValueCell[ind]))

                if is_global:
                    local_descriptor = normalizeDescriptor(local_descriptor, bins);
                    descriptor.extend(local_descriptor)
                else:
                    descriptor.extend(local_descriptor)

    if is_global:
        descriptorglobal = normalizeDescriptorGlobal(descriptor)
        return descriptorglobal
    else:
        return descriptor


def computePHOGLAB(Img, angle, bins, levels, section):
    
    GradientX, GradientY = maxGradient_fast(Img)
    
    # Calculate the norm (strength) of Gradient values
    GradientValue = np.sqrt( np.square(GradientX) + np.square(GradientY) )
    
    # Replace zeros in GradientX with a small value to avoid Zero division
    GradientX[np.where(GradientX == 0)] = 1e-5
    
    YX = GradientY / GradientX

    if angle == 180:
        GradientAngle = ((np.arctan(YX) + (np.pi / 2)) * 180) / np.pi
    elif angle == 360:
        GradientAngle = ((np.arctan2(GradientY, GradientX) + np.pi) * 180) / np.pi
    else:
        raise ValueError("Invalid angle value. Use 180 or 360.")
        
    descriptor = computeDescriptor(GradientValue, GradientAngle, bins, angle, levels, section)
    
    return descriptor, GradientValue, GradientAngle
    

def convert_to_matlab_lab(img_rgb):
    '''
    Matlab has a different range for the channels of the LAB color space.
    We need to scale to the Matlab ranges, to get the same results
    '''
    
    img = color.rgb2lab(img_rgb) 

    #L: 0 to 100, a: -127 to 128, b: -128 to 127
    img[:,:,0] = np.round(np.array((img[:,:,0] / 100) * 255 )).astype(np.int32)
    img[:,:,1] = np.round(np.array(img[:,:,1] + 128).astype(np.int32))
    img[:,:,2]= np.round(np.array(img[:,:,2] + 128).astype(np.int32))
    
    return img.astype(np.uint16)
    

def normalizeDescriptor(descriptor, bins):
    b = np.reshape(descriptor, (bins, len(descriptor) // bins), order='F')
    c = np.sum(b, axis=0)
    s = b.shape
    
    temp = np.zeros((s[0], s[1]))

    for i in range(s[1]):
        if c[i] != 0:
            temp[:, i] = b[:, i] / c[i]
        else:
            temp[:, i] = b[:, i]
    
    normalizeddescriptor = np.reshape(temp, len(descriptor), order='F')
    return list(normalizeddescriptor)


def normalizeDescriptorGlobal(descriptor):
    if np.sum(descriptor) != 0:
        normalizeddescriptorGlobal = descriptor / np.sum(descriptor)
        return normalizeddescriptorGlobal
    else:
        return list(descriptor)


def computeWeightedDistances(descriptor, bins, levels, section, descriptornn):
    distances = []

    comparisonglobal = descriptor[:bins]

    temp = np.zeros((levels, 2), dtype=int)

    temp[0, 0] = bins + 1
    temp[0, 1] = section ** (2) * bins + temp[0, 0] - 1

    for i in range(1, levels):
        temp[i, 0] = temp[i - 1, 1] + 1
        temp[i, 1] = section ** ((i+1) * 2) * bins + temp[i, 0] - 1

    distances.append(np.sum(comparisonglobal))

    for i in range(levels):
        for j in range(temp[i, 0], temp[i, 1] + 1, bins):
            j = j-1
            part = descriptor[j:j + bins]
            
            if (np.max(comparisonglobal) > 1e-8) and (np.max(part) > 1e-8):
                dist1 = np.sum(np.minimum(comparisonglobal, part))

                m1 = np.mean(descriptornn[:bins])
                m2 = np.mean(descriptornn[j:j + bins])

                area = section ** ((i+1) * 2)
                m2 = m2 * area

                if m1 < 1e-8 or m2 < 1e-8:
                    strengthsimilarity = 0
                elif m1 > m2:
                    strengthsimilarity = m2 / m1
                else:
                    strengthsimilarity = m1 / m2

                dist1 = dist1 * strengthsimilarity
                distances.append(dist1)
            else:
                distances.append(0)

    return distances

def computeSD(descriptorglobal, bins, levels, section):
    temp = np.zeros((levels, 2), dtype=int)
    temp[0, 0] = bins + 1
    temp[0, 1] = section ** (2) * bins + temp[0, 0] - 1
    
    for i in range(1, levels):
        temp[i, 0] = temp[i - 1, 1] + 1
        temp[i, 1] = section ** ((i+1) * 2) * bins + temp[i, 0] - 1
        
    descript = descriptorglobal[temp[levels - 1, 0]-1 : temp[levels - 1, 1]]
    sdvalue = np.std(descript)
    return sdvalue


def displayDistances(distances, bins, levels, section):
    distanceatlevel = []
    
    temp3 = np.zeros([levels+1, 2], dtype=int)
    # print('###########', temp3, levels)
    temp3[0, 0] = 1
    temp3[0, 1] = 1

    for i in range(levels):
        temp3[i+1, 0] = section ** (2 * (i+1))
        temp3[i+1, 1] = temp3[i+1, 0] + temp3[i, 1]

    distanceatlevel = np.median(distances[temp3[levels - 1, 1] :temp3[levels, 1]])

    return distanceatlevel



def PHOGfromImage(img_rgb, section=2, bins=16, angle=360, levels=3, re=None, sesfweight=[1,1,1] ):
    
    img = resize_img(img_rgb, re)
    img = convert_to_matlab_lab(img)

    descriptor, GradientValue , GradientAngle = computePHOGLAB(img, angle, bins, levels, section)
    descriptornn = descriptor

    descriptor=normalizeDescriptor(descriptor,bins)

    descriptorglobal = computeDescriptor(GradientValue, GradientAngle, bins, angle, levels, section, is_global=True)
    
    distances=computeWeightedDistances(descriptor,bins,levels,section,descriptornn);

    anisotropy = computeSD(descriptorglobal, bins, levels, section)
    complexity = np.mean(GradientValue)

    # self_sim
    distancesatlevel=0;
    distancesateachlevel = []
    for i in range(levels):
       distancesateachlevel.append(displayDistances(distances,bins,i+1,section));
       distancesatlevel=distancesatlevel+distancesateachlevel[i]*sesfweight[i];
    self_sim=distancesatlevel/sum(sesfweight);      
        
    return self_sim, complexity, anisotropy




########################################################################################################################
################################# Huebner Group ########################################
########################################################################################################################


def APB_Score(im):
    '''
    1. nall sollte Zahl der Scharzen Pixel sein, ist hier aber die Summer der Intensitätswerte im Bild
    2. Im Matlab code wird der treshold wert 126 in zwei Schleifen addiert!!! Hier auch so übernommen, ist aber eigentlich ein Fehler.
    '''
    
    height, width = im.shape

    hist = np.histogram(im, bins=256, range=(0, 256))

    counts = hist[0]
    
    thres = 128

    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres-1:])  # hier Fehler in Berechnungen, aber kaum Auswirkunge auf Ergebnisse
    
    if sum1 <= sum2:
        im_comp = 255 - im  # Invert image
    else:
        im_comp = im

    nall = np.sum(im_comp)   ### Number of Pixels with value of 0
    
    #print('NALL:', nall)
    
    ## to avoid division 0
    if nall == 0:
        nall = 1

    # Horizontal balance
    w = width // 2
    


    s1 = np.sum(im_comp[:, :w], dtype=int)
    s2 = np.sum(im_comp[:, -w:], dtype=int)
    bh = (abs(s1 - s2) / nall) * 100  
        
    w2 = width // 4 # adding center row of w to middle area if w uneven 

    s1 = np.sum(im_comp[:, :w2], dtype=int)
    s2 = np.sum(im_comp[:, -w2:], dtype=int)

    bioh = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100 # %  inner-outer horizontal 

    # Vertical balance
    h = height // 2

    s1 = np.sum(im_comp[:h, :], dtype=int)
    s2 = np.sum(im_comp[-h:, :], dtype=int)
    bv = (abs(s1 - s2) / nall) * 100
    
    h2 = height // 4
    s1 = np.sum(im_comp[:h2, :], dtype=int)
    s2 = np.sum(im_comp[-h2:, :], dtype=int)

    biov = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Main diagonal and inner-outer (bottom right top left)
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bmd = (abs(s1 - s2) / nall) * 100

    prop = 1 / np.sqrt(2)
    b1 = height - int(height * prop)
    b2 = width - int(width * prop)
    s1 = np.sum(np.tril(im_comp, -b1), dtype=int)
    s2 = np.sum(np.triu(im_comp, b2), dtype=int)
    biomd = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Anti-diagonal and inner-outer (bottom right top left)
    im_comp = np.rot90(im_comp)
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bad = (abs(s1 - s2) / nall) * 100

    s1 = np.sum(np.tril(im_comp, -b2), dtype=int)
    s2 = np.sum(np.triu(im_comp, b1), dtype=int)
    bioad = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    bs = (bh + bv + bioh + biov + bmd + biomd + bad + bioad) / 8

    return bs




def DCM_Key(im):
    '''
    Input grayscale in range [0-256]
    
    '''
    
    height, width = im.shape

    hist = np.histogram(im, bins=256, range=(0, 256))
    counts = hist[0]
       
    thres = 128

    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres:])
    

    if sum1 <= sum2:
        im_comp = 255 - im  # Invert image
        inv = 'inverted';
    else:
        im_comp = im
        inv = 'original';

    nall = np.sum(im_comp)   ### Number of Pixels with value of 0
    
    # Horizontal balance point
    r = 0
    for i in range(width):
        w = np.sum(im_comp[:, i],dtype=float)
        r += w * i
    Rh = np.round(r / nall) + 1  # x position of fulcrum
    Rhnorm = Rh / width  # Normalized
    

    # Vertical balance point
    r = 0
    for i in range(height):
        w = np.sum(im_comp[i, :],dtype=float)
        r += w * i
    Rv = np.round(r / nall) + 1  # y position of fulcrum
    Rvnorm = Rv / height  # Normalized

    htmp = 0.5 - Rhnorm
    vtmp = 0.5 - Rvnorm

    dist = np.sqrt(htmp ** 2 + vtmp ** 2)
    rdist = (dist / 0.5) * 100

    # Calculate direction in degrees
    xcenter = width / 2
    ycenter = height / 2

    # Direction goes from 0 to 180° (upper half, right to left), from 0 to -180° lower half (right to left)
    direction = np.degrees(np.arctan2(ycenter - Rv, Rh - xcenter))

    if (direction >= 45) and (direction < 135):
        area = 1  # top
    elif (direction >= 135) or (direction < -135):
        area = 2  # left
    elif (direction >= -135) and (direction < -45):
        area = 3  # bottom
    else:
        area = 4  # right

    return rdist


def MS_Score(img_gray):

    # Automatically find optimal threshold level
    level  = threshold_otsu(img_gray)

    # Convert image to binary
    BW = img_gray <= level

    s = BW.shape
    height = s[0]
    width = s[1]

    # Horizontal axis of reflection (vertical reflection)
    if height % 2 == 0:  # even number
        h2 = height // 2
    else:
        h2 = (height - 1) // 2
    n1 = h2 - 1 # why?
    
    
    sym = 0
    for i in range(width):
        for j in range(h2):
            #print(i,j, sym)
            sym += (BW[j, i] * BW[ (height-1) - (j), i]) * (1 + j / n1)
    Sh = sym * (2 / (3 * width * h2))

    # Vertical axis of reflection (horizontal reflection)
    if width % 2 == 0:  # even number
        w2 = width // 2
    else:
        w2 = (width - 1) // 2
    n1 = w2 - 1
    
    sym = 0
    for i in range(height):
        for j in range(w2):
            sym += (BW[i, j] * BW[i, (width-1) - j]) * (1 + j / n1)
    Sv = sym * (2 / (3 * height * w2))

    if width == height:
        # Major diagonal of reflection (ONLY FOR SQUARES)
        sym = 0
        n = 1  # Pixels until diagonal
        for i in range(1,height):
            for j in range(n):
                #print(i,j,n)
                sym += (BW[i, j] * BW[j, i]) * (1 + (j+1) / n)
            n += 1
            
        Smd = sym * (2 / (3 * height * (width - 1) / 2))
            
        # Minor diagonal of reflection (ONLY FOR SQUARES)
        BW = rotate(BW, 90)
        sym = 0
        n = 1  # Pixels until diagonal
        for i in range(1, height):
            for j in range(n):
                sym += (BW[i, j] * BW[j, i]) * (1 + (j+1) / n)
            n += 1
        Sad = sym * (2 / (3 * height * (width - 1) / 2))

        ms = ((Sh + Sv + Smd + Sad) / 4) * 100
    else:
        ms = ((Sh + Sv) / 2) * 100

    return ms


def entropy_score_2d(im):  ## takes grayscale image!!

    # number of bins taken from original paper: Hübner & Fillinger. Comparison of Objective Measures for Predicting Perceptual Balance and Visual Aesthetic Preference. Page: 4
    hbins = 10;
    vbins = 10;
    
    height, width = im.shape
    
    hist = np.histogram(im, bins=256, range=(0, 256))
    counts = hist[0]
    thres = 128
    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres-1:])  # hier Fehler in Berechnungen, aber kaum Auswirkunge auf Ergebnisse
    if sum1 <= sum2:
        im = 255 - im  # Invert image
        print('inverted')
    else:
        im = im
    
    
    level  = threshold_otsu(im)
    
    BW = im > level

    hinc = width // hbins
    vinc = height // vbins
    
    x = np.zeros((vbins, hbins))
    
    ## summing up black pixels in cells
    for i in range(hbins+1):
        for j in range(vbins+1):
            if (i!=hbins) and (j!=vbins): # inner pieces
                x[j,i] = np.sum( BW[ vinc * j : (j+1) * vinc ,  hinc*i : hinc * (i+1)     ]    )
            elif (i<hbins) and (j==vbins): # residuals vertical
                x[j-1,i]   += np.sum( BW[ vinc * j :  ,  hinc*i : hinc * (i+1)     ]    )
            elif (i==hbins) and (j<vbins): # residuals horizontal
                x[j,i-1]   += np.sum( BW[ vinc * j : (j+1) * vinc ,  hinc*i :      ]    )
            elif (i==hbins) and (j==vbins): # residuals horizontal
                x[j-1,i-1] += np.sum( BW[ vinc * j :  ,  hinc*i :     ]    )
    
    nbins = hbins * vbins
    max_entropy = np.log2(nbins)
    
    xh = x.flatten(order='F')
    all_sum = np.sum(xh)
    
    #j = np.nonzero(xh)
    # # entropy
    # xh = xh / all_sum    
    # hx = -np.sum(xh[j] * np.log2(xh[j]))
    # en_all = (hx / max_entropy) * 100

    # horizontal entropy
    max_entropy = np.log2(hbins)   
    y = np.sum(x, axis=0) / all_sum
    j = np.nonzero(y)
    en = -np.sum(y[j] * np.log2(y[j]))
    en_hori = (en / max_entropy) * 100
    
    #vertical entropy
    max_entropy = np.log2(vbins)
    y = np.sum(x.T, axis=0) / all_sum
    j = np.nonzero(y)
    en = -np.sum(y[j] * np.log2(y[j]))
    en_vert = (en / max_entropy) * 100

    # average of hori and vert entropy
    en_av = (en_hori + en_vert) / 2
    
    return en_av



########################################################################################################################
########################################   BOX Count Algos #############################################################
########################################################################################################################



### Branka Spehar, "2D" Box Count Algo using binary images
def box_count_2d(im):

    nr, nc = im.shape  # get y & x dimensions of image

    # Scale image to square if not already square
    if nr > nc:
        im = np.asarray(PIL.Image.fromarray(im).resize((nc, nc)))
    elif nr < nc:
        im = np.asarray(PIL.Image.fromarray(im).resize((nr, nr)))

    threshold = np.mean(im)
    b = im < threshold
    b = b.astype(np.uint8)

    x, y = [], []
    i = 1
    while b.shape[0] > 6:
        x.append(b.shape[0])
        y.append(np.sum(b))

        c = np.zeros((b.shape[0]//2, b.shape[1]//2))
        for xx in range(c.shape[0]):
            for yy in range(c.shape[1]):

                c[xx, yy] = np.sum (  b[xx*2  : xx*2 + 2  , yy*2 : yy*2 + 2 ]  )

        b = (c > 0) & (c < 4)
        i += 1

    params = np.polyfit(np.log2(x[1:]), np.log2(y[1:]), 1)
    D = params[0]
    return D




### George Mather, "3D" Box Count Algo unsing graylevels
def custom_differential_box_count(img_gray):  ### Takes 2 dim Image/array as input

    ### center crop largest rectangle image
    nr, nc = img_gray.shape
    # Find largest square that is a power of 2 (for box count)
    if nr < nc:
        nk = 2**np.ceil(np.log2(nr))
        nnr = 2**(np.ceil(np.log2(nr)) - 1) if nr < nk else 2**np.ceil(np.log2(nr))
        dc = nc - nnr
        dr = nr - nnr
        nnc = nnr
    else:
        nk = 2**np.ceil(np.log2(nc))
        nnc = 2**(np.ceil(np.log2(nc)) - 1) if nc < nk else 2**np.ceil(np.log2(nc))
        dr = nr - nnc
        dc = nc - nnc
        nnr = nnc
    nr, nc = int(nnr), int(nnc)
    # Centred crop
    I = img_gray[int(round(dr / 2)):int(round(dr / 2 + nr)), int(round( dc / 2)):int(round(dc / 2 + nc))]


    ### calc box counts
    nr, nc = I.shape
    # Calculate min and max box sizes
    minpow = int(np.ceil(np.log2(nr**(1/3))))

    bmin = 2**minpow
    bmax = bmin
        
    while np.ceil(nr / bmax + 1) <= np.ceil(nr / (bmax - 1)):
        bmax = bmax + 1
        
    boxes = np.arange(bmin, bmax + 1, 2)

    boxHeights = boxes * (1 / nr)  # box size in greylevels

    boxCounts = np.zeros(len(boxes))
    boxSizes = np.zeros(len(boxes))

    # loop through the box sizes
    for b in range(len(boxes)):
        
        bs = boxes[b]
        bh = boxHeights[b]  # box size in graylevels

        # Divide the image into a grid of boxes (bs x bs).
        # Loop through the cells in the grid, calculating the box count for
        # each and adding it to the running total.
        # Overlap columns by one x-pixel
        boxCount = 0
        for by in range(1, nc - bs, bs):
            
            for bx in range(1, nr - bs + 1, bs-1):
                submat = I[by-1: by + bs - 1 , bx-1 : bx + bs - 1 ]
                l = np.max(submat)
                k = np.min(submat)

                if l == k:
                    b1 = 1
                else:
                    b1 = np.ceil((l - k) / bh)
                boxCount = boxCount + b1

        # Now use the range of box sizes to calculate D
        boxCounts[b] = boxCount
        boxSizes[b] = 1.0 / bs


    dfit = np.polyfit(np.log(boxSizes), np.log(boxCounts), 1)
    D = dfit[0]
    return D  
    



############################################################################################################################
################################################## Unity Tests ######################################################
############################################################################################################################


if __name__ == "__main__":
    
        
    # rootdir_img = '/home/ralf/Documents/18_SIP_Machine/test_img/img/Arcimboldo3.jpg'
    
    
    rootdir_img = '/home/ralf/Documents/18_SIP_Machine/test_img/img/Aivazovsky1.jpg'
    
    
    
    img = np.array( PIL.Image.open(  rootdir_img   ))
    
    img_rgb = np.asarray(PIL.Image.open(rootdir_img).convert('RGB'))
               
    img_lab = color.rgb2lab(img_rgb)
    img_hsv = color.rgb2hsv(img_rgb)
    
    
    print('Lab(b):', mean_channels(img_lab)[2])
    print('HSV(S):', mean_channels(img_hsv)[0])
    print('HSV(H)entropy:', shannonentropy_channels(img_hsv)[0])
    
    print('H circmean and circstd:' , circ_stats(img_hsv) )
    
    # ### test mean channel values
    # r,g,b = mean_channels(img)
    # assert(pytest.approx(r)==94.0273279507882)
    # assert(pytest.approx(g)==80.3288427527874)
    # assert(pytest.approx(b)==49.7325951557093)
    
    # # test std channel values
    # std_r, std_g, std_b = STD_channels(img)
    # assert(pytest.approx(std_r)==54.71982792778597)
    # assert(pytest.approx(std_g)==53.09180131856316)
    # assert(pytest.approx(std_b)==40.28622896197172)
    
    
    
    
    # ## load test csv from old code
    # df_test = pd.read_csv(  '/home/ralf/Documents/18_SIP_Machine/test_img/test_img.csv' , sep=',')
    # df_test = df_test[df_test.img_name == 'Arcimboldo3.jpg'].copy()
    # # df_sel = df_t[df_t.category == 'JA(aesth)'].copy()
    # # df_test.drop(['score','scaled_score'], axis=1, inplace=True)
    
    
    
    # res_dict = {}
    
    
    # # ## test CNN-measures
    # img = np.array( PIL.Image.open(  rootdir_img   )  , dtype= np.float32  )[:,:,(2,1,0)]
    # [kernel,bias] = np.load(open("/home/ralf/Documents/18_SIP_Machine/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)
    # resp_scipy = conv2d(img, kernel, bias)
    # _, normalized_max_pooling_map_p12 = max_pooling (resp_scipy, patches=12 )
    # _, normalized_max_pooling_map_p22 = max_pooling (resp_scipy, patches=22 )
    
    
    # res_dict['variability'] = get_CNN_Variance (normalized_max_pooling_map_p12, kind='variability' )
    # res_dict['sparseness'] = get_CNN_Variance (normalized_max_pooling_map_p22, kind='sparseness' )
    
    
    # # ## test Symmetry
    # lr,ud,lrud = get_symmetry(img, kernel, bias)
    # res_dict['symmetry-lr'] = lr
    # res_dict['symmetry-ud'] = ud
    # res_dict['symmetry-lrud'] = lrud
    
    
    # # ### 
    
    
    # # img = np.array( PIL.Image.open(  rootdir_img   ).convert('L'))
    # # print(do_first_and_second_order_entropy(img))
    


    # for i in res_dict:
    #     print (i, res_dict[i])
    
    # # for entry in df_test.iterrows():
    # #     print(entry)
    
    
    
    
    
    
    
    
    
    


