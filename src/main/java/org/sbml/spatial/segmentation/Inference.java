package org.sbml.spatial.segmentation;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The class Inference.
 * 
 * This class allows model inference by taking an unsegmented microscopic
 * cellular image as input and it returns a segmented image. Date Created:
 * August 1, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */
public class Inference {
	/**
	 * ComputationGraph model the deep learning model (U-Net)
	 */
	private ComputationGraph model;
	/**
	 * ZooModel model2 the deep learning model (U-Net) to be used when setting 1
	 * input channel
	 */
	private ZooModel model2;
	/**
	 * String path the path for input image
	 */
	private String path;
	/**
	 * String directory the path for working directory
	 */
	private String directory;
	/**
	 * int Width the width of input image set to 128
	 */
	public static final int WIDTH = 128;
	/**
	 * int HEIGHT the height of input image set to 128
	 */
	public static final int HEIGHT = 128;
	/**
	 * int Channels the number of input channels in image set to 3
	 */
	public static final int CHANNELS = 3;

	/**
	 * 
	 * Constructor for class Inference.java. Takes model, path and directory as
	 * input.
	 * 
	 * @param model     the ComputationGraph Model
	 * @param path      the string for path of the target unsegmented image
	 * @param directory the string for working directory
	 */

	public Inference(ComputationGraph model, String path, String directory) {
		this.model = model;
		this.path = path;
		this.directory = directory;
	}

	public Inference(ZooModel model, String path, String directory) {
		this.model2 = model;
		this.path = path;
		this.directory = directory;
	}

	// For inferring the model

	/**
	 * The method which returns inferred image.
	 * 
	 * @return bufferedImage the inferred image of type BufferedImage
	 */

	public BufferedImage imgOut() {
		BufferedImage bufferedImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
		;
		try {
			NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
			ImageType img = new ImageType(new File(path));
			BufferedImage bufferedBGR = img.getBGRBufferedImage();
			INDArray imageNative = loader.asMatrix(bufferedBGR);
			imageNative.shapeInfoToString();
			imageNative = imageNative.reshape(1, CHANNELS, HEIGHT, WIDTH);
			imageNative = imageNative.divi(255f);
			INDArray[] output = model.output(imageNative);
			for (INDArray out : output) {
				out = out.reshape(1, HEIGHT, WIDTH);
				// out = out.permute(2,1,0);
				// bufferedImage = new BufferedImage(WIDTH, HEIGHT,
				// BufferedImage.TYPE_BYTE_GRAY);
				for (int i = 0; i < WIDTH; i++) {
					for (int j = 0; j < HEIGHT; j++) {
						float f = out.getFloat(new int[] { 0, j, i });
						int gray = (int) (f * 255.0);
						if (gray > 127) {
							bufferedImage.setRGB(i, j, Color.WHITE.getRGB());
						} else {
							bufferedImage.setRGB(i, j, Color.BLACK.getRGB());
						}
					}
				}

				// ImageIO.write(bufferedImage, "tif", new File(directory + File.separator +
				// "outputUnet.tif"));
			}

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}

		return bufferedImage;
	}

}
