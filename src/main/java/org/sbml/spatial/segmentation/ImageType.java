package org.sbml.spatial.segmentation;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ImageType {

	private File imageFile;

	public ImageType(File imageFile) {
		this.imageFile = imageFile;
	}

	// RGB TO BGR for enhanced performance
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

	public BufferedImage getBGRBufferedImage() {
		try {
			BufferedImage bufferedImage = ImageIO.read(this.imageFile);
			return getBGRBufferedImage(bufferedImage);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

}
