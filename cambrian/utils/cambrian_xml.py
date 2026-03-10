"""This module provides a helper class for manipulating mujoco xml files. It provides
some helper methods that wrap the `xml` library. This is useful for modifying mujoco xml
files in a programmatic way. MjSpec has since provided functionality directly in
Mujoco to support this functionality."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, TypeAlias

import mujoco as mj
from hydra_config import HydraContainerConfig

from cambrian.utils.logger import get_logger

MjCambrianXMLConfig: TypeAlias = HydraContainerConfig | List[Dict[str, Self]]
"""
Actual type: List[Dict[str, Self]]
OmegaConf complains since we have a Dict inside a List

We use a list here because then we can have non-unique keys.

This defines a custom XML config. This can be used to define custom XMLs which
are built from during the initialization phase of the environment. The config is
structured as follows:

.. code-block:: yaml

    - parent_key1:
        - child_key1:
            - attr1: val1
            - attr2: val2
        - child_key2:
            - attr1: val1
            - attr2: val2
    - child_key1:
        - child_key2:
            - attr1: ${parent_key1.child_key1.attr2}
    - child_key2:
        - child_key3: ${parent_key1.child_key1}

which will construct an XML that looks like:

.. code-block:: xml

    <parent_key1>
        <child_key1 attr1="val1" attr2="val2">
            <child_key2 attr1="val2"/>
        </child_key1>
        <child_key2>
            <attr1>val1</attr1>
            <attr2>val2</attr2>
            <child_key3 attr1="val1" attr2="val2">
        </child_key2>
    </parent_key1>

This is a verbose representation for XML files. This is done
to allow interpolation through Hydra/OmegaConf in the XML files and without the need
for a complex XML parser OmegaConf resolver.
"""


class MjCambrianXML:
    """Helper class for manipulating mujoco xml files. Provides some helper methods for
    that wrap the `xml` library.

    Args:
        base_xml_path (Path | str): The path to the base xml file to load.

    Keyword Args:
        overrides (Optional[MjCambrianXMLConfig]): The xml config to override the base
            xml file with. This is a list of dictionaries. See `MjCambrianXMLConfig`
            for more information.
    """

    WHITELIST_ATTRIBUTES = ["name"]
    WHITELIST_TAGS = ["global", "include", "scale", "map", "headlight"]

    def __init__(
        self,
        base_xml_path: Path | str,
        *,
        overrides: Optional[MjCambrianXMLConfig] = None,
    ):
        self._base_xml_path = Path(base_xml_path)

        self.load(self._base_xml_path)
        if overrides is not None:
            self += MjCambrianXML.from_config(overrides)

        # Post process by combining the root with itself.
        # All duplicate elements will be combined.
        self += self

    def load(self, path: Path | str):
        """Load the xml from a file."""
        assert Path(path).exists(), f"File does not exist: {path}"
        self._tree = ET.parse(path)
        self._root = self._tree.getroot()

    def write(self, path: Path | str):
        """Write the xml to a file. Will pretty write the xml."""
        xml_string = self.to_string()
        with open(path, "w") as f:
            f.write(xml_string)

    @staticmethod
    def make_empty() -> "MjCambrianXML":
        """Loads an empty mujoco xml file. Only has the `mujoco` and worldbody
        `tags`."""
        return MjCambrianXML.from_string("<mujoco><worldbody></worldbody></mujoco>")

    @staticmethod
    def from_string(xml_string: str) -> "MjCambrianXML":
        """Loads the xml from a string."""
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(xml_string)
            f.flush()
            return MjCambrianXML(f.name)

    @staticmethod
    def from_config(config: MjCambrianXMLConfig) -> "MjCambrianXML":
        """Adds to the xml based on the passed config.

        The MjCambrianXMLConfig is structured as follows:

        .. code-block:: yaml

            - parent_key:
                - child_key:
                    - attribute_key: attribute_value
                    - subchild_key:
                        - attribute_key: attribute_value
                        - attribute_key: attribute_value
                    - subchild_key:
                        - subsubchild_key: attribute_value

        This would create the following xml:

        .. code-block:: xml

            <parent_key>
                <child_key attribute_key="attribute_value">
                    <subchild_key attribute_key="attribute_value"
                        attribute_key="attribute_value"/>
                    <subchild_key>
                        <subsubchild_key attribute_key="attribute_value"/>
                    </subchild_key>
                </child_key>
            </parent_key>

        We need to walk though the XMLConfig and update it recursively.

        See `MjCambrianXMLConfig` for more information.
        """

        xml = MjCambrianXML.make_empty()

        def add_element(parent: ET.Element, tag: str, **kwargs) -> ET.Element:
            # If the parent is the root, just return
            # NOTE Only going to compare the tag, not the attributes
            if tag == xml._root.tag:
                return parent

            # Make the search path
            path = tag + "".join(
                [f"[@{key}='{value}']" for key, value in kwargs.items()]
            )
            if (element := parent.find(path)) is None:
                element = xml.add(parent, tag, **kwargs)
            return element

        def get_attribs(
            config: MjCambrianXMLConfig, *, depth: int = 0
        ) -> Dict[str, str]:
            attribs = {}
            for key, value in config.items():
                if isinstance(value, str):
                    attribs[key] = value
                elif depth == 0 and isinstance(value, list):
                    # If it's a list, we need to add a new element for each item in
                    # the list
                    for sub_config in value:
                        attribs.update(get_attribs(sub_config, depth=depth + 1))
            return attribs

        def add_to_xml(parent: ET.Element, config: Dict[str, MjCambrianXMLConfig]):
            for key, value in config.items():
                if isinstance(value, list):
                    attribs = get_attribs(config)
                    for sub_config in value:
                        # If it's a list, we need to add a new element for each item in
                        # the list
                        element = add_element(parent, key, **attribs)
                        add_to_xml(element, sub_config)
                else:
                    # If it's a value, we need to add it as an attribute
                    parent.set(key, str(value))

        for root in config:
            add_to_xml(xml.root, root)
        return xml

    @staticmethod
    def parse(
        xml_string: str, *, overrides: Optional[MjCambrianXMLConfig] = None
    ) -> str:
        """This is a helper method to parse an xml file with overrides."""
        xml = MjCambrianXML.from_string(xml_string)
        if overrides is not None:
            xml += MjCambrianXML.from_config(overrides)
        return xml.to_string()

    def add(self, parent: ET.Element, tag: str, *args, **kwargs) -> ET.Element:
        """Add an element to the xml tree.

        Args:
            parent (ET.Element): The parent element to add the new element to.
            tag (str): The tag of the new element.
            *args: The arguments to pass to the `ET.SubElement` call.
            **kwargs: The keyword arguments to pass to the `ET.SubElement` call.
        """
        return ET.SubElement(parent, tag, *args, **kwargs)

    def remove(self, parent: ET.Element, element: ET.Element):
        """Remove an element from the xml tree.

        Args:
            parent (ET.Element): The parent element to remove the element from.
            element (ET.Element): The element to remove.
        """
        parent.remove(element)

    def find(
        self, tag: str, *, _all: bool = False, **kwargs
    ) -> List[ET.Element] | ET.Element | None:
        """Find an element by tag.

        If any additional keyword arguments are passed, they will be used to filter the
        elements, as in the element must have the attribute and it must be equal to the
        value. In this case, `ET.iterfind` will be used. It uses the predicates
        described here: https://docs.python.org/3/library/xml.etree.elementtree.html\
            #supported-xpath-syntax.

        If no keyword arguments are passed, `ET.find` will be used.

        If not found, `None` will be returned.

        Args:
            tag (str): The tag of the element to find.
            _all (bool): If true, will use `ET.findall`, else `ET.find`.
            **kwargs: The keyword arguments to filter the elements by.

        Returns:
            List[ET.Element] | ET.Element | None: The element or `None` if not found. If
                `_all` is true, a list of elements will be returned.
        """
        kwargs_str = "".join([f"[@{key}='{value}']" for key, value in kwargs.items()])
        if _all:
            return self._root.findall(f"{tag}{kwargs_str}")
        else:
            return self._root.find(f"{tag}{kwargs_str}")

    def findall(self, tag: str, **kwargs) -> List[ET.Element]:
        """Alias for `find(tag, _all=True, **kwargs)`."""
        return self.find(tag, _all=True, **kwargs)

    def get_path(self, element: ET.Element) -> Tuple[List[ET.Element], str]:
        """Get the path of an element in the xml tree. Unfortunately, there is no
        built-in way to do this. We'll just iterate up the tree and build the path.

        Args:
            element (ET.Element): The element to get the path to.

        Returns:
            Tuple[List[ET.Element], str]: The list of elements in the path and the
            string representation of the path. The list of elements is ordered from root
            down to the passed element. NOTE: the root is _not_ included in the list.
        """

        path: List[str] = []
        elements: List[ET.Element] = []

        parent_map = {child: parent for parent in self._root.iter() for child in parent}
        while element is not None:
            if element == self._root:
                break

            path.insert(0, element.tag)
            elements.insert(0, element)
            element = parent_map.get(element)

        return elements, "/".join(path)

    def combine(self, root: ET.Element, other: ET.Element) -> ET.Element:
        """Combines two xml trees. Preferred to be called through the `+` or `+=`
        operators.

        Taken from here: https://stackoverflow.com/a/29896847/20125256
        """

        class hashabledict(dict):
            def __hash__(self):
                return hash(tuple(sorted(self.items())))

        def create_key(el: ET.Element):
            for attribute in self.WHITELIST_ATTRIBUTES:
                if attribute in el.attrib:
                    return (el.tag, el.attrib[attribute])
            for tag in self.WHITELIST_TAGS:
                if el.tag == tag:
                    return (el.tag, "")
            return (el.tag, hashabledict(el.attrib))

        # Create a mapping from tag name to element, as that's what we are filtering
        # with
        mapping = {create_key(el): el for el in root}
        for el in other:
            key = create_key(el)
            if len(el) == 0:
                # Not nested
                try:
                    # Update the text
                    mapping[key].text = el.text
                    # Merge attributes
                    mapping[key].attrib.update(el.attrib)
                except KeyError:
                    # An element with this name is not in the mapping
                    mapping[key] = el
                    # Add it
                    root.append(el)
            else:
                try:
                    # Recursively process the element, and update it in the same way
                    self.combine(mapping[key], el)
                except KeyError:
                    # Not in the mapping
                    mapping[key] = el
                    # Just add it
                    root.append(el)
        return root

    @property
    def root(self) -> ET.Element:
        """The root element of the xml tree."""
        return self._root

    @property
    def base_dir(self) -> Path:
        """The directory of the base xml file."""
        return self._base_xml_path.parent

    def __add__(self, other: Self) -> Self:
        assert isinstance(other, MjCambrianXML)
        self += other
        return self

    def __iadd__(self, other: Self) -> Self:
        assert isinstance(other, MjCambrianXML)
        self._tree = ET.ElementTree(self.combine(self._root, other._root))
        return self

    def to_string(self) -> str:
        """This pretty prints the xml to a string. toprettyxml adds a newline at the end
        of the string, so we'll remove any empty lines."""
        import xml.dom.minidom as minidom

        string = ET.tostring(self._root, encoding="unicode")
        string = minidom.parseString(string).toprettyxml(indent=" ")
        return "\n".join([line for line in string.split("\n") if line.strip()])

    def __str__(self) -> str:
        return self.to_string()

    def to_spec(self) -> mj.MjSpec:
        """Convert the xml to a mujoco spec."""
        return mj.MjSpec.from_string(self.to_string())


def load_xml(input_xml_file: str) -> MjCambrianXML:
    return MjCambrianXML(input_xml_file)


def convert_xml_to_yaml(
    base_xml_path: str, *, overrides: Optional[MjCambrianXMLConfig] = None
) -> str:
    """This is a helper method to convert an xml file to a yaml file.
    This is for loading and overriding an xml file from the yaml config files. We have
    to convert the xml to yaml in order to apply interpolations in the yaml file. Like,
    to support ${parent:xml} (which evaluates the xml parent name) in the xml, we have
    to convert the xml to yaml and then apply the interpolations."""

    xml = MjCambrianXML(base_xml_path, overrides=overrides)
    root = xml.root

    def parse_element(element: ET.Element, result: list) -> list:
        # Set's attributes
        for key, value in element.items():
            result.append({key: value})

        # Adds children
        for child in element:
            result.append({child.tag: parse_element(child, [])})
        return result

    return parse_element(root, [])


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="MujocoXML Tester")

    parser.add_argument(
        "--convert-yaml", type=str, help="Convert a yaml to xml", default=None
    )
    parser.add_argument("--load-xml", type=str, help="Load an xml file", default=None)

    args = parser.parse_args()

    if args.convert_yaml is not None:
        with open(args.convert_yaml, "r") as f:
            config = yaml.safe_load(f)
        xml = MjCambrianXML.from_config(config)
        get_logger().info(xml)
        exit()
    if args.load_xml is not None:
        xml = load_xml(args.load_xml)
        get_logger().info(xml)
        exit()

    xml = MjCambrianXML.make_empty()

    # NOTE: This and the next example fail because combine doesn't combine attributes
    # Unclear whether this is an issue or not
    config = [{"mujoco": [{"compiler": [{"angle": "degree"}]}]}]
    xml += MjCambrianXML.from_config(config)

    config = [{"mujoco": [{"compiler": [{"coordinate": "local"}]}]}]
    xml += MjCambrianXML.from_config(config)
    get_logger().info(xml)

    config = """
    - mujoco:
        - model: ant
        - custom:
            - numeric:
                - name: init_qpos_${..name}
                - data: 0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0
        - asset:
            - material:
                - name: geom_mat
                - emission: '0.1'
                - rgba: 0.8 0.6 0.4 1
        - default:
            - default:
                - class: ant
                - joint:
                    - limited: 'true'
                    - armature: '1'
                    - damping: '1'
                - geom:
                    - condim: '3'
                    - conaffinity: '0'
                    - margin: '0.01'
                    - friction: 1 0.5 0.5
                    - solref: .02 1
                    - solimp: .8 .8 .01
                    - density: '5.0'
                    - material: geom_mat
        - actuator:
            - motor:
                - ctrllimited: 'true'
                - ctrlrange: -1 1
                - gear: '150.0'
                - joint: motor1_rot_{uid}
            - motor:
                - ctrllimited: 'true'
                - ctrlrange: -1 1
                - gear: '150.0'
                - joint: motor2_rot_{uid}
    - mujoco:
        - compiler:
            - coordinate: global
            - angle: radian
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    xml += MjCambrianXML.from_config(config)

    get_logger().info(xml)
