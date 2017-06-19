/***
 * Class Job3SortingComparator
 * Job3 Sorting Comparator class
 * @author sgarouachi
 */

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class Job3SortingComparator extends WritableComparator {

	protected Job3SortingComparator() {
		super(FloatWritable.class, true);
	}
	
	/**
	 * Compare two float DESC
	 * 
	 * @return Int -1 0 1
	 */
	@Override
	public int compare(WritableComparable w1, WritableComparable w2) {
		Float w1_float = Float.parseFloat(w1.toString());
		Float w2_float = Float.parseFloat(w2.toString());
		return -w1_float.compareTo(w2_float);
	}
}