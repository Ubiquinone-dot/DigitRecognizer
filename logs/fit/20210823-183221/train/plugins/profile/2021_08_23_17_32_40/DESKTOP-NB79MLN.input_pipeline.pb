	?$??c3@?$??c3@!?$??c3@	????V<??????V<??!????V<??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?$??c3@?lV}????AL?
F%E3@YHP?s??rEagerKernelExecute 0*	fffffFT@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr??????!9g?#-?E@)?X?? ??1?q?jB@:Preprocessing2U
Iterator::Model::ParallelMapV2?]K?=??!?&?Ff0@)?]K?=??1?&?Ff0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvq?-??!?qEl{3@)'???????1E???c*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[?????!X5?v?R@)HP?sׂ?1?%?Z#?&@:Preprocessing2F
Iterator::ModelQ?|a2??!?*??'?9@)???_vO~?15?d??"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?HP?x?!???@)?HP?x?1???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??ZӼ?t?!y?u'@)??ZӼ?t?1y?u'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????V<??I??????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?lV}?????lV}????!?lV}????      ??!       "      ??!       *      ??!       2	L?
F%E3@L?
F%E3@!L?
F%E3@:      ??!       B      ??!       J	HP?s??HP?s??!HP?s??R      ??!       Z	HP?s??HP?s??!HP?s??b      ??!       JCPU_ONLYY????V<??b q??????X@