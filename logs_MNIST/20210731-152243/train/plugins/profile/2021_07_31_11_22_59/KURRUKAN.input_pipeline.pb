	̶??ca@̶??ca@!̶??ca@	@P@?;AC@@P@?;AC@!@P@?;AC@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:̶??ca@?y?9[@??A??R%?UU@Y??o?N?J@rEagerKernelExecute 0*	?????+?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?yƾd???!?u???F<@)?Ӟ?sb??1?T?f??8@:Preprocessing2U
Iterator::Model::ParallelMapV2??+H3??!???_AK8@)??+H3??1???_AK8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[z4Փ???!??????9@)??????10??M?}2@:Preprocessing2F
Iterator::Model??\???!15 [v?C@)?]?)ʥ??1O???V?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?K??????!q?? C@)?K??????1q?? C@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????6??!?????(N@)?8?d?˙?1?j?ΏV@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?1=a???!?YZz@)?1=a???1?YZz@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapmT?YO??!$???xt;@)#?-?R\??1z?Xp?^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 38.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9@P@?;AC@I???}ľN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?y?9[@???y?9[@??!?y?9[@??      ??!       "      ??!       *      ??!       2	??R%?UU@??R%?UU@!??R%?UU@:      ??!       B      ??!       J	??o?N?J@??o?N?J@!??o?N?J@R      ??!       Z	??o?N?J@??o?N?J@!??o?N?J@b      ??!       JCPU_ONLYY@P@?;AC@b q???}ľN@