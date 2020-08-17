package org.sbml.spatial.segmentation;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Inference {
	private ComputationGraph model;
	// private ZooModel model;
	private String path;
	private String directory;
	public static final int WIDTH = 128;
	public static final int HEIGHT = 128;
	public static final int CHANNELS = 3;

	public Inference(ComputationGraph model, String path, String directory) {
		this.model = model;
		this.path = path;
		this.directory = directory;
	}

	/*
	 * public Inference(ZooModel model, String path, String directory) { this.model
	 * = model; this.path = path; this.directory = directory;}
	 */

	// For inferring the model
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
