"""Summarize precision properties from pytest --junitxml (including failures)."""
import argparse
import json
import xml.etree.ElementTree as ET


def summarize(path, scope="all", property_name="precision"):
    result = {"cases":0,"failed_cases":[],"criteria_failures":{},"worst":{}}
    for case in ET.parse(path).iter("testcase"):
        for prop in case.findall("./properties/property"):
            if prop.get("name") != property_name:
                continue
            report = json.loads(prop.get("value"))
            if scope=="bounded" and not report["within_tau_bound"]:
                continue
            if scope=="stress" and report["within_tau_bound"]:
                continue
            result["cases"] += 1
            name = case.get("name")
            if case.find("failure") is not None:
                result["failed_cases"].append(name)
            for key,value in report.items():
                if key.endswith("_passes") and not value:
                    result["criteria_failures"].setdefault(key,[]).append(name)
            groups = ("same_bf16_inputs","fp32_interpolation","interpolation_quantization")
            if property_name=="embedding_precision":
                groups = tuple(k for k,v in report.items() if isinstance(v,dict))
            elif "same_bf16_inputs" not in report:
                groups = ("chain",)
            for group in groups:
                quantities = report if group=="chain" else report[group]
                if "max_abs" in quantities:
                    quantities = {"tensor":quantities}
                for quantity,values in quantities.items():
                    if not isinstance(values,dict):
                        continue
                    for metric,value in values.items():
                        key = f"{group}.{quantity}.{metric}"
                        previous = result["worst"].get(key)
                        worse = previous is None or (value<previous["value"] if metric.startswith("cosine") else value>previous["value"])
                        if worse:
                            result["worst"][key] = {"value":value,"case":name}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit_xml")
    parser.add_argument("--scope",choices=("all","bounded","stress"),default="all")
    parser.add_argument("--property",choices=("precision","embedding_precision"),default="precision")
    args = parser.parse_args()
    print(json.dumps(summarize(args.junit_xml,args.scope,args.property),indent=2))
