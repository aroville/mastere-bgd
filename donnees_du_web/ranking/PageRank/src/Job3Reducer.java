import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

//Only used for Unit Tests
public class Job3Reducer extends
		Reducer<FloatWritable, Text, FloatWritable, Text> {
	@Override
	public void reduce(FloatWritable key, Iterable<Text> values, Context context)
			throws InterruptedException, IOException {
        List<String> list = new ArrayList<>();
		for (Text value : values)
		    list.add(value.toString());

        Collections.sort(list);
        for (String t: list)
            context.write(key, new Text(t));
	}
}