	gDio??A@gDio??A@!gDio??A@	i??,???i??,???!i??,???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:gDio??A@5^?IB@A?HP?x?@YQ?|a??rEagerKernelExecute 0*	23333?U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB>?٬???!5??EM@@)?D???J??1,ӕ??t<@:Preprocessing2U
Iterator::Model::ParallelMapV2HP?sג?!??te?25@)HP?sג?1??te?25@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??@??ǈ?!O???S?+@)??@??ǈ?1O???S?+@:Preprocessing2F
Iterator::Model???_vO??!?3??A@)Ǻ?????1?9??s?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0?*??!N?&w??7@)?? ?rh??1MW?+ӕ#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???QI??!???yP@)?HP?x?1Aq?8P@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}m?!\.?@)??H?}m?1\.?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?+e?X??!?وlD:@)/n??b?1"???F@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9i??,???I???i?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5^?IB@5^?IB@!5^?IB@      ??!       "      ??!       *      ??!       2	?HP?x?@?HP?x?@!?HP?x?@:      ??!       B      ??!       J	Q?|a??Q?|a??!Q?|a??R      ??!       Z	Q?|a??Q?|a??!Q?|a??b      ??!       JCPU_ONLYYi??,???b q???i?X@