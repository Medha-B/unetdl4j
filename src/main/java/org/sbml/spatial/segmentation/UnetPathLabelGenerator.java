package org.sbml.spatial.segmentation;

import java.io.IOException;
import java.net.URI;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;

/**
 * The class UnetPathLabelGenerator.
 * 
 * This class is used for obtaining the label of a file directly from the given
 * path. Date Created: July 4, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */

public class UnetPathLabelGenerator implements PathLabelGenerator {
	// private NativeImageLoader imageLoader = new NativeImageLoader(256,256,1);
	private NativeImageLoader imageLoader = new NativeImageLoader(128, 128, 1);

	@Override
	public Writable getLabelForPath(String path) {
		try {
			String newPath = new String(path);
			return new NDArrayWritable(imageLoader.asMatrix(newPath.replace("image", "gt")).divi(255.0f));
		} catch (IOException ioe) {
			ioe.printStackTrace();
			return null;
		}
	}

	@Override
	public Writable getLabelForPath(URI uri) {
		return getLabelForPath(uri.getPath());
	}

	@Override
	public boolean inferLabelClasses() {
		return false;
	}
}
