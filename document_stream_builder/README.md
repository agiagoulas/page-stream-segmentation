# Document Stream Builder

Python script to build <b>consecutive document streams</b> from a collection of pdf documents.

## Usage

```console
python document_stream_builder.py ||
    --input <Input Dir>           ||
    --output <Output Dir>         ||
    --random <True/False>         ||  
    --limit <Number>
```

- input: Input directory (Default: "./input/")
- output: Output directory (Default: "./output/")
- random: Random document order in page stream (Default: True)
- limit: limit the amount of processed input documents