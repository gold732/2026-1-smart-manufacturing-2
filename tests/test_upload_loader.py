import io
import unittest
from unittest.mock import patch

import pandas as pd
from upload_loader import UploadError, load_uploaded_data


def upload(content, name):
    stream = io.BytesIO(content)
    stream.name = name
    return stream


class UploadLoaderTests(unittest.TestCase):
    def test_csv_bom_uppercase_and_rewound_stream(self):
        stream = upload('그룹,값\na,12\n'.encode('utf-8-sig'), 'data.CSV')
        stream.read()
        result = load_uploaded_data(stream)
        self.assertEqual(result.columns.tolist(), ['그룹', '값'])
        self.assertEqual(result.iloc[0].tolist(), ['a', 12])

    def test_real_xlsx(self):
        stream = upload(b'', 'data.XLSX')
        expected = pd.DataFrame({'group': ['a', 'b'], 'value': [1, 2]})
        expected.to_excel(stream, index=False, engine='openpyxl')
        pd.testing.assert_frame_equal(load_uploaded_data(stream), expected)

    def test_invalid_inputs_have_actionable_errors(self):
        cases = [
            (b'a,b\n1,2\n', 'data.txt', '지원하지 않는'),
            (b'a,b\n1,2\n', 'data', '지원하지 않는'),
            (b'', 'data.csv', '비어'),
            (b'a,b\n', 'data.csv', '데이터 행'),
            (b'a,b\n,\n', 'data.csv', '데이터 행'),
            ('그룹,값\na,1'.encode('cp949'), 'data.csv', 'UTF-8'),
            (b'a,b\n1,2\n3,4,5\n', 'data.csv', '열 개수'),
            (b'not an excel file', 'data.xlsx', '실제 XLSX'),
        ]
        for content, name, message in cases:
            with self.subTest(name=name, content=content):
                with self.assertRaisesRegex(UploadError, message):
                    load_uploaded_data(upload(content, name))

    def test_empty_excel(self):
        stream = upload(b'', 'empty.xlsx')
        pd.DataFrame(columns=['a', 'b']).to_excel(stream, index=False)
        with self.assertRaisesRegex(UploadError, '데이터 행'):
            load_uploaded_data(stream)

    def test_excel_dependency_and_read_errors(self):
        for error, message in [(ImportError('missing'), 'openpyxl'),
                               (ValueError('invalid workbook'), '암호 보호'),
                               (OSError('read error'), '읽을 수 없습니다')]:
            with self.subTest(error=error):
                with patch('upload_loader.pd.read_excel', side_effect=error):
                    with self.assertRaisesRegex(UploadError, message):
                        load_uploaded_data(upload(b'file', 'data.xlsx'))

    def test_app_stops_after_upload_error(self):
        # Execute the upload block only; no Streamlit server or analysis tabs needed.
        import ast
        from pathlib import Path
        from unittest.mock import Mock
        tree = ast.parse(Path('app.py').read_text())
        block = next(node for node in tree.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'uploaded_file is not None')
        class StopExecution(Exception):
            pass
        ui = Mock()
        ui.stop.side_effect = StopExecution
        context = dict(st=ui, uploaded_file=upload(b'broken', 'bad.xlsx'),
                       load_uploaded_data=load_uploaded_data, UploadError=UploadError)
        with self.assertRaises(StopExecution):
            exec(compile(ast.Module(body=[block], type_ignores=[]), 'app.py', 'exec'), context)
        self.assertIn('실제 XLSX', ui.error.call_args.args[0])
        self.assertNotIn('df_raw', context)


if __name__ == '__main__':
    unittest.main()
