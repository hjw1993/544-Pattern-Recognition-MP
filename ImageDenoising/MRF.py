import pylab as pl
import numpy as np
import argparse


def MRF(I, J, eta, zeta, beta):
    """ 
    Perform Inference to determine the label of every pixel.
    1. Go through the image in random order.
    2. Evaluate the energy for every pixel being being set to either 1 or -1.
    3. Assign the label resulting in the least energy to the pixel in question.

    Inputs: 
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.   
    Output:
        denoised_image: The denoised image after one MRF iteration.   
    """
    m, n = I.shape
    denoised_image=np.zeros((m,n))
    x_idx = np.arange(len(I))
    np.random.shuffle(x_idx)
    y_idx = np.arange(len(I[0]))
    np.random.shuffle(y_idx)
    for i in range(len(x_idx)):
        for j in range(len(y_idx)):
            pos_en=energy_evaluation(I,J,i,j,1,eta,zeta,beta)
            neg_en=energy_evaluation(I, J, i, j, -1, eta, zeta, beta)
            if(pos_en<neg_en):
                denoised_image[i][j]=1
            else:
                denoised_image[i][j]=-1





    return denoised_image
 

def energy_evaluation(I, J, pixel_x_coordinate, pixel_y_coordinate, 
    pixel_value, eta, zeta, beta):
    """
    Evaluate the energy of the image of a particular pixel set to either 1or-1.
    1. Set pixel(pixel_x_coordinate,pixel_y_coordinate) to pixel_value
    2. Compute the unary, and pairwise potentials.
    3. Compute the energy

    Inputs: 
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        pixel_x_coordinate: the x coordinate of the pixel to be evaluated.
        pixel_y_coordinate: the y coordinate of the pixel to be evaluated.
        pixel_value: could be 1 or -1.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.   
    Output:
        energy: Energy value.

    """
    i=pixel_x_coordinate
    j=pixel_y_coordinate
    neighbor=0
    M=len(I)
    N=len(I[0])
    if (i == 0 and j == 0):
        neighbor = zeta*pixel_value*(J[i+1][j]+J[i][j+1])
    elif i == 0 and j == N - 1:
        neighbor = zeta * pixel_value * (J[i + 1][j] + J[i][j - 1])
    elif i == M - 1 and j == 0:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i][j + 1])
    elif i == M - 1 and j == N - 1:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i][j - 1])
    elif i == 0:
        neighbor = zeta * pixel_value * (J[i + 1][j] + J[i][j + 1]+J[i][j - 1])
    elif i == M - 1:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i][j + 1]+J[i][j - 1])
    elif j == 0:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i + 1][j] + J[i][j + 1])
    elif j == N - 1:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i + 1][j] + J[i][j - 1])
    else:
        neighbor = zeta * pixel_value * (J[i - 1][j] + J[i + 1][j] + J[i][j - 1]+J[i][j + 1])

    energy=beta*pixel_value-eta*pixel_value*I[pixel_x_coordinate][pixel_y_coordinate]-neighbor
    return energy


def greedy_search(noisy_image, eta, zeta, beta, conv_margin):
    """
    While convergence is not achieved (this verified by calling 
    the function 'not_converged'),
    1. iteratively call the MRF function to perform inference.
    2. If the number of iterations is above 10, stop and return 
    the image that you have at the 10th iteration.

    Inputs: 
        noisy_image: the noisy image.
        eta: Weight of the pairwise observed-latent potential.
        zeta: Weight of the pairwise latent-latent potential.
        beta: Weight of unary term.    
        conv_margin: Convergence margin
    Output:
        denoised_image: The denoised image.   
    """
    
    # Noisy Image.
    I = noisy_image.copy()
    image_new=I
    image_old=I

    for i in range(10):
        image_new=MRF(I,image_old,eta,zeta,beta)
        if(not_converged(image_new,image_old,i,conv_margin)):
            break
        image_old=image_new
    denoised_image=image_new
    return denoised_image


def not_converged(image_old, image_new, itr, conv_margin):
    """
    Check for convergence. Convergence is achieved if the denoised image 
    does not change between two consequtive iterations by a certain 
    margin 'conv_margin'.
    1. Compute the percentage of pixels that changed between two
     consecutive iterations.
    2. Convergence is achieved if the computed percentage is below 
    the convergence margin.

    Inputs:
        image_old: Denoised image from the previous iteration.
        image_new: Denoised image from the current iteration.
        itr: The number of iteration.
        conv_margin: Convergence margin.
    Output:  
        has_converged: a boolean being true in case of convergence
    """

    has_converged=(1-np.sum(np.abs(image_old-image_new))/(image_old.size*2)>=conv_margin)
    return has_converged


def load_image(input_file_path, binarization_threshold):
    """
    Load image and binarize it by:
    0. Read the image.
    1. Consider the first channel in the image
    2. Binarize the pixel values to {-1,1} by setting the values 
    below the binarization_threshold to -1 and above to 1.
    Inputs: 
        input_file_path.
        binarization_threshold.
    Output: 
        I: binarized image.
    """

    img=pl.imread(input_file_path)


    img=np.array(img)

    I=img[:,:,0]


    for i in range(len(I)):
        for j in range(len(I[0])):
             if I[i][j]>binarization_threshold:
                 I[i][j]=1.

             else:
                 I[i][j]=-1.


    return I 


def inject_noise(image):
    """
    Inject noise by flipping the value of some randomly chosen pixels.
    1. Generate a matrix of probabilities of every pixel 
    to keep its original value .
    2. Flip the pixels if its corresponding probability in 
    the matrix is below 0.1.

    Input:
        image: original image
    Output:
        noisy_image: Noisy image
    """
    noisy_image = image.copy()
    m,n=image.shape
    pro=np.random.rand(m,n)

    for i in range(len(image)):
        for j in range(len(image[0])):
             if pro[i][j]<0.1:
                noisy_image[i][j]*=-1.




    return noisy_image


def f_reconstruction_error(original_image, reconstructed_image):
    """
    Compute the reconstruction error (L2 loss)
    inputs:
        original_image.
        reconstructed_image.
    output: 
        reconstruction_error: MSE of reconstruction.
    """
    m,n=original_image.shape
    reconstruction_error =0
    for i in range(m):
        for j in range(n):
            reconstruction_error+=np.power((original_image[i][j]-reconstructed_image[i][j]),2)
    reconstruction_error/=(m*n)

    return reconstruction_error


def plot_image(image, title, path):
    pl.figure()
    pl.imshow(image)
    pl.title(title)
    pl.savefig(path)


def parse_arguments(parser):
    """
    Parse arguments from the command line
    Inputs: 
        parser object
    Output:
        Parsed arguments
    """
    parser.add_argument('--input_file_path',
        type=str,
        default="img/seven.png",
        metavar='<input_file_path>', 
        help='Path to the input file.')

    parser.add_argument('--weight_pairwise_observed_unobserved',
        type=float,
        default=2,
        metavar= '<weight_pairwise_observed_unobserved>',
        help='Weight of observed-unobserved pairwise potential.')

    parser.add_argument('--weight_pairwise_unobserved_unobserved',
        type=float, 
        default=1.5,
        metavar='<weight_pairwise_unobserved_unobserved>',
        help='Weight of unobserved-unobserved pairwise potential.')

    parser.add_argument('--weight_unary', 
        type=float, 
        default=0.1, 
        metavar='<weight_unary>', 
        help='Weight of the unary term.')

    parser.add_argument('--convergence_margin', 
        type=float, 
        default=0.999, 
        metavar='<convergence_margin>', 
        help='Convergence margin.')

    parser.add_argument('--binarization_threshold', 
        type=float,\
        default=0.05, \
        metavar='<convergence_margin>',
        help='Perc of different btw the images between two iter.')
    
    args = parser.parse_args()
    return args


def main(): 
    # Read the input arguments
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # Parse the MRF hyperparameters
    eta = args.weight_pairwise_observed_unobserved
    zeta = args.weight_pairwise_unobserved_unobserved
    beta = args.weight_unary

    # Parse the convergence margin
    conv_margin = args.convergence_margin  

    # Parse the input file path
    input_file_path = args.input_file_path
     
    # Load the image.  
    I = load_image(input_file_path, args.binarization_threshold)

    # Create a noisy version of the image.
    J = inject_noise(I)
    
    # Call the greedy search function to perform MRF inference
    newJ = greedy_search(J, eta, zeta, beta, conv_margin)

    # Plot the Original Image
    plot_image(I, 'Original Image', 'img/Original_Image')

    # Plot the Denoised Image
    plot_image(newJ, 'Denoised version', 'img/Denoised_Image')
    
    # Compute the reconstruction error 
    reconstruction_error = f_reconstruction_error(I, newJ)
    print('Reconstruction Error: ', reconstruction_error)

    
if __name__ == "__main__":
    main()