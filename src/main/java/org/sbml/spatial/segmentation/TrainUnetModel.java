package org.sbml.spatial.segmentation;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainUnetModel {

	private static final Logger log = LoggerFactory.getLogger(TrainUnetModel.class);
	private static final int WIDTH = 128;
	private static final int HEIGHT = 128;
	// private static final int CHANNELS = 3;
	private static final int CHANNELS = 1; // for 1 input channel

	/**
	 * Performs inline replacement RGB -> BGR for memory performance
	 *
	 * @param bufferedImage
	 * @return
	 */
	public static BufferedImage getBGRBufferedImage(BufferedImage bufferedImage) {
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

	public static BufferedImage getBGRBufferedImage(File imageFile) {
		try {
			BufferedImage bufferedImage = ImageIO.read(imageFile);
			return getBGRBufferedImage(bufferedImage);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");

			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			String pathToImage;
			if (args.length > 0) {
				pathToImage = args[0];
			} else {
				pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
			}

			DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
			UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
			Random rng = new Random();
			FileSplit inputSplit = new FileSplit(rootDir, allowedExtensions, rng);
			ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labeler);
			imageRecordReader.initialize(inputSplit);
			int labelIndex = 1;
			DataSetIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize,
					labelIndex, labelIndex, true);
			scaler.fit(imageDataSetIterator);
			imageDataSetIterator.setPreProcessor(scaler);

			UIServer uiServer = UIServer.getInstance();

			// StatsStorage statsStorage = new FileStatsStorage(new
			// File("C:\\Users\\Subroto\\StatsLog"));

			// uiServer.attach(statsStorage);

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
			// learningScheduleMap.put(1000, 0.00001);

			int numEpochs = 1;

			// ComputationGraph model = UNet.builder().updater(new
			// Adam(1e-4)).build().init();
			// ComputationGraph model = UNet.builder().updater(new Adam(new
			// MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();

			// To change the number of input channels from 3 to 1
			ZooModel unet = UNet.builder().updater(new Adam(1e-4)).build();
			unet.setInputShape(new int[][] { { 1, 128, 128 } });
			ComputationGraph model = (ComputationGraph) unet.init();

			StatsStorage ss = new InMemoryStatsStorage();
			uiServer.attach(ss);
			model.addListeners(new ScoreIterationListener(), new StatsListener(ss));
			model.fit(imageDataSetIterator, numEpochs);

			log.warn(model.summary());

			NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
			BufferedImage bufferedBGR = getBGRBufferedImage(new File(pathToImage));
			INDArray imageNative = loader.asMatrix(bufferedBGR);

			log.warn(imageNative.shapeInfoToString());

			imageNative = imageNative.reshape(1, CHANNELS, HEIGHT, WIDTH);
			imageNative = imageNative.divi(255f);

			INDArray[] output = model.output(imageNative);
			// INDArray sigmoid = Transforms.sigmoid(output);
			for (INDArray out : output) {
				out = out.reshape(1, HEIGHT, WIDTH);
				// out = out.permute(2,1,0);
				BufferedImage bufferedImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
				for (int i = 0; i < WIDTH; i++) {
					for (int j = 0; j < HEIGHT; j++) {
						float f = out.getFloat(new int[] { 0, j, i });
						int gray = (int) (f * 255.0);
						// if (gray > 127) {
						// bufferedImage.setRGB(i,j,Color.WHITE.getRGB());
						// } else {
						// bufferedImage.setRGB(i,j,Color.BLACK.getRGB());
						// }
						bufferedImage.setRGB(i, j, new Color(gray, gray, gray).getRGB());
					}
				}
				ImageIO.write(bufferedImage, "tif", new File(directory + File.separator + "outputUnet.tif"));
				// ImageIO.write(bufferedImage,"tif",new File(home + File.separator +
				// "outputUnet.tif"));
			}
		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
