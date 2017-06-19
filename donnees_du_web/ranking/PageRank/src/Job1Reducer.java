/***
 * Class Job1Reducer
 * Job1 Reducer class
 * @author sgarouachi
 */

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class Job1Reducer extends Reducer<Text, Text, Text, Text> {

	/**
	 * Job1 Reduce method (page, 1.0 \t outLinks)
	 * Remove redundant links & sort them Asc
	 */
	@Override
	public void reduce(Text key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
        Set<String> set = new HashSet<>();
        for (Text t : values)
            set.add(t.toString());

        List<String> list = new ArrayList<>(set);
        Collections.sort(list);
        String s = list.isEmpty() ? "" : "\t"+String.join(",", list);
        context.write(key, new Text("1.0"+s));
	}
}
