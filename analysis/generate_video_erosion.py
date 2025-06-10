import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

def pad_image(image, pad_value=0):
    # Calculate the padding required to make the dimensions divisible by 16
    height, width, _ = image.shape
    pad_height = (16 - (height % 16)) % 16
    pad_width = (16 - (width % 16)) % 16
    
    # Pad the image
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
    
    return padded_image

def create_phi_video(input_file, output_file):
    # Load the saved numpy array
    data = np.load(input_file)

    phi_array = data["phi"]

    # Get the number of steps and the shape of the flux grid
    num_steps = phi_array.shape[0]
    height, width = phi_array.shape[1:3]

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # List to store images in memory
    images = []

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    # Add colors for over and under values
    cmap.set_under('dimgray')

    # Normalize with boundaries
    clip = 1.0
    norm = mcolors.Normalize(vmin=0, vmax=clip, clip=False) #

    # Iterate over each simulation step
    for step in range(num_steps):
        phi_grid = phi_array[step]

        # Create a plot
        fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8))
        ax.imshow(phi_grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax.axis('off')  # Turn off the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding
        
        # Draw the canvas and convert to an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, 1:]  # Drop alpha if needed
        plt.close(fig)
        
        # Pad the image to make dimensions divisible by 16
        padded_image = np.rot90(pad_image(image), k=1)
        images.append(padded_image)
    
    # Create the video
    imageio.mimsave(output_file, images, fps=24, format='FFMPEG')
    print(f'Video saved as {output_file}')

def create_flux_video(input_file, output_file, vmin, vmax):
    # Load the saved numpy array
    data = np.load(input_file)

    flux_array = data["flux"]

    # Get the number of steps and the shape of the flux grid
    num_steps = flux_array.shape[0]
    height, width = flux_array.shape[1:3]

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # List to store images in memory
    images = []

    # Create a colormap
    cmap = plt.get_cmap('magma')

    # Add colors for over and under values
    cmap.set_under('k')

    # Normalize with boundaries
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)#, clip=False) #
#    norm = mcolors.Normalize(vmin=0, vmax=2)#, clip=False) #


    # Iterate over each simulation step
    for step in range(num_steps):

        flux_grid = flux_array[step]
        logflux = np.log10(flux_grid, out=vmin*np.ones_like(flux_grid, dtype=np.float64), where=(flux_grid!=0))

        # Create a plot
        fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8))
        ax.imshow(logflux, cmap=cmap, norm=norm, interpolation='nearest')
#        ax.imshow(flux_grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax.axis('off')  # Turn off the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding
        
        # Draw the canvas and convert to an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, 1:]  # Drop alpha if needed
        plt.close(fig)
        
        # Pad the image to make dimensions divisible by 16
        padded_image = np.rot90(pad_image(image), k=1)
        images.append(padded_image)
    
    # Create the video
    imageio.mimsave(output_file, images, fps=24, format='FFMPEG')
    print(f'Video saved as {output_file}')



# Load data from the .npz file
config = "linsweep_202505261456"
Q = 0.5
T = 10

input_file = f'./DATA/{config}/F_{Q:.1f}_T_{T:.1f}_V_001/data.npz'

#input_file = f'./DATA/erosion/{config}/F_{Q:g}_T_{T:.1f}_V_001/data.npz'

output_file = f'./Results/{config}_E_F_{Q:g}_T_{T:g}_V_001_phi.mp4'
create_phi_video(input_file, output_file)

#output_file = f'./Results/{config}_E_F_{Q:g}_T_{T:g}_V_001_flux.mp4'
#create_flux_video(input_file, output_file,vmin=-3,vmax=0.3)


# data = np.load(input_file)

# savior = data["array"]

# # List to store the frames
# images = []

# # Iterate over the arrays in 'savior'
# for i, phi_array in enumerate(savior):
#     if i % 1 == 0:  # No need for np.mod, use Python's modulo
#         plt.figure()
#         plt.imshow(np.rot90(phi_array), vmin=0, vmax=1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.gca().set_xticks([])  # Remove x-ticks
#         plt.gca().set_yticks([])  # Remove y-ticks

#         # Save the current figure as an image in memory
#         plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
#         plt.close()

#         # Read the saved image using ImageIO v3 syntax and append to list
#         images.append(iio.imread('temp_frame.png'))

# # Save the list of images as a video using ImageIO v3
# iio.imwrite(output_file, images, fps=5)
