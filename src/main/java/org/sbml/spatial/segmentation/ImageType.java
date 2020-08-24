package org.sbml.spatial.segmentation;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * The class ImageType.
 * 
 * This class implements inference image formatting for the deep-learning model
 * U-Net. Date Created: August 1, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */
public class ImageType {

	/**
	 * File imageFile the input image file
	 */
	private File imageFile;

	/**
	 * Constructor for class ImageType.java. Takes the target image file as
	 * parameter.
	 * 
	 * @param imageFile which is the target image file.
	 */

	public ImageType(File imageFile) {
		this.imageFile = imageFile;
	}

	/**
	 * For conversion from RGB to BGR for enhanced memory performance
	 * 
	 * @param bufferedImage which is of type BufferedImage
	 * @return bufferedImage which is of type BufferedImage after processing it
	 */
	public BufferedImage getBGRBufferedImage(BufferedImage bufferedImage) {
		for (int w = 0; w < bufferedImage.getWidth(); w++) {
			for (int h = 0; h < bufferedImage.getHeight(); h++) {
				int p = bufferedImage.getRGB(w, h);
				int a = (p >> 24) & 0xff;
				int r = (p >> 16) & 0xff;
				int g = (p >> 8) & 0xff;
				int b = p & 0xff;
				// swap r (red) and b (blue) channels
				p = (a << 24) | (b << 16) | (g << 8) | r;
				bufferedImage.setRGB(w, h, p);
			}
		}
		return bufferedImage;
	}

	/**
	 * Calls the getBGRBufferedImage(BufferedImage bufferedImage) function after
	 * reading the image file. Used for handling exceptions.
	 * 
	 * @return bufferedImage the BufferedImage after processing it
	 */
	public BufferedImage getBGRBufferedImage() {
		try {
			BufferedImage bufferedImage = ImageIO.read(this.imageFile);
			return getBGRBufferedImage(bufferedImage);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

}
